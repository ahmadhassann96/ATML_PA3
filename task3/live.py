"""Live comparison of parallel sampling and sequential refinement (PA3 Task 3, step 4).

    python -m task3.live --export                 # locally: write the 50 questions to outputs/task3/live/questions.json
    python -m task3.live --shard 0 --n_shards 2   # on a GPU (Kaggle T4): generate, appending to calls_shard0.jsonl
    python -m task3.live --smoke --model Qwen/Qwen2.5-0.5B-Instruct   # quick local check

Qwen/Qwen2.5-3B-Instruct at the pinned revision, FP16, batch size 1, temperature 0.7, top-p 0.8, top-k 20,
repetition penalty 1.0, native chat template with the system message "You are a helpful assistant." and
the original problem followed by two newlines and the step-by-step/boxed instruction. Per question:
* single:   one response capped at 3,072 new tokens
* parallel: three independent responses capped at 1,024 tokens each (self-consistency picks the answer)
* refine:   the first parallel response is the initial answer; one critique-and-revision response capped
            at 2,048 tokens, conditioned on it (assistant turn) and the fixed review instruction
Every call has its own seed (6304 * 1000 + 10 * question index + call index). Prompts, responses, seeds,
token counts, cap hits, extracted answers, wall-clock time (excluding model loading) and peak GPU
memory are saved per call. Re-running resumes after the last completed call.
"""
import argparse
import time

import torch

from common.utils import append_jsonl, env_info, load_json, read_jsonl, save_json
from task3 import config as C

CALLS = [("single", 0, C.LIVE_LIMITS["single"])] + \
        [("parallel", k, C.LIVE_LIMITS["parallel"]) for k in range(C.LIVE_LIMITS["n_parallel"])] + \
        [("refine", 0, C.LIVE_LIMITS["revision"])]


def live_dir():
    d = C.task_dir() / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_boxed(text):
    """Last \\boxed{...} with balanced braces (supplied README rule); None if absent, empty or unclosed."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    if depth:
        return None
    ans = text[j:k - 1].strip()
    return ans or None


def export():
    import pandas as pd
    ids = load_json(C.task_dir() / "splits" / "math.json")["live50"]
    g = pd.read_parquet(C.cache_dir() / "math_rollouts.parquet", columns=["sample_id", "question_text", "gold_answer"])
    g = g.drop_duplicates("sample_id").set_index("sample_id")
    qs = [dict(index=i, sample_id=s, problem=g.loc[s, "question_text"], gold=g.loc[s, "gold_answer"])
          for i, s in enumerate(ids)]
    save_json(dict(rule="first 50 MATH evaluation-split questions in split order", questions=qs),
              live_dir() / "questions.json")
    print(f"exported {len(qs)} questions")


def generate(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qs = load_json(live_dir() / "questions.json")["questions"]
    model_id, rev = (args.model, None) if args.model else (C.LIVE_MODEL, C.LIVE_REVISION)
    if args.smoke:
        qs = qs[:2]
    out = live_dir() / (f"calls_smoke{args.shard}.jsonl" if args.smoke else f"calls_shard{args.shard}.jsonl")
    done = {(r["index"], r["strategy"], r["call"]) for r in read_jsonl(out)} if out.exists() else set()
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    load_s = time.perf_counter() - t0
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    save_json(dict(env=env_info(), model=model_id, revision=rev, dtype="float16", batch_size=1, smoke=args.smoke,
                   gen_kwargs=C.GEN_KW, limits=C.LIVE_LIMITS, load_time_s=load_s, eos_token_ids=sorted(eos)),
              live_dir() / (f"run_info_smoke{args.shard}.json" if args.smoke else f"run_info_shard{args.shard}.json"))

    def call(messages, max_new, seed):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, **C.GEN_KW)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        new = gen[0, ids.shape[1]:].tolist()
        stopped = any(x in eos for x in new)
        n_out = len(new)
        body = new[:next((i for i, x in enumerate(new) if x in eos), len(new))]
        text = tok.decode(body, skip_special_tokens=True)
        return dict(prompt=prompt, response=text, seed=seed, n_input_tokens=int(ids.shape[1]),
                    n_output_tokens=n_out, n_response_tokens=len(body), hit_cap=bool(not stopped and n_out >= max_new),
                    max_new_tokens=max_new, extracted=extract_boxed(text), time_s=dt,
                    peak_gpu_mem_bytes=int(torch.cuda.max_memory_allocated()))

    for q in qs:
        if q["index"] % args.n_shards != args.shard:
            continue
        base = [{"role": "system", "content": C.SYSTEM_MSG},
                {"role": "user", "content": q["problem"] + "\n\n" + C.INSTRUCTION}]
        initial = None
        for ci, (strategy, k, max_new) in enumerate(CALLS):
            key = (q["index"], strategy, k)
            if strategy == "parallel" and k == 0 and key in done:
                initial = [r for r in read_jsonl(out) if r["index"] == q["index"] and r["strategy"] == "parallel"
                           and r["call"] == 0][0]["response"]
            if key in done:
                continue
            seed = C.SEED * 1000 + 10 * q["index"] + ci
            if strategy == "refine":
                msgs = base + [{"role": "assistant", "content": initial}, {"role": "user", "content": C.REVIEW_MSG}]
            else:
                msgs = base
            r = call(msgs, max_new // 16 if args.smoke else max_new, seed)
            if strategy == "parallel" and k == 0:
                initial = r["response"]
            append_jsonl(dict(index=q["index"], sample_id=q["sample_id"], strategy=strategy, call=k, **r), out)
            print(f"[shard {args.shard}] q{q['index']:02d} {strategy}[{k}] out={r['n_output_tokens']} "
                  f"cap={r['hit_cap']} {r['time_s']:.1f}s ans={str(r['extracted'])[:30]!r}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="2 questions, token caps / 16, separate output files")
    ap.add_argument("--model", default=None, help="override the model (smoke tests only)")
    a = ap.parse_args()
    if a.export:
        export()
    else:
        generate(a)
