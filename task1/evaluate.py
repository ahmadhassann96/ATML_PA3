"""Evaluate one of the seven conditions with lm-evaluation-harness (FP16, dropout off, zero-shot).

  GSM8K   : gsm8k_cot, num_fewshot=0, non-thinking chat template, greedy, 1,024 new tokens,
            stock stop sequences and answer filters (primary: flexible-extract exact match)
  HellaSwag / BoolQ / ARC-Easy : stock completion prompts, no chat template, candidate likelihood

    python -m task1.evaluate --condition baseline
    python -m task1.evaluate --condition lora16 --limit 20      # quick check on a few docs

Writes outputs/task1/eval/<condition>/{summary.json, gsm8k_samples.jsonl, <task>_samples.jsonl}
"""
import argparse
import time

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from common.utils import env_info, save_json, write_jsonl
from task1 import config as C
from task1.modeling import condition_spec, load_condition, load_tokenizer, merge_check


def _seed_kwargs():
    return dict(random_seed=C.SEED, numpy_random_seed=C.SEED, torch_random_seed=C.SEED, fewshot_random_seed=C.SEED)


def gsm8k_rows(samples, tok):
    rows = {}
    for s in samples:
        d = rows.setdefault(s["doc_id"], dict(doc_id=s["doc_id"], question=s["doc"]["question"],
                                              target=s["target"], response=s["resps"][0][0],
                                              prompt=s["arguments"][0][0] if s.get("arguments") else None))
        d[f"extracted_{s['filter']}"] = s["filtered_resps"][0]
        d[f"correct_{s['filter']}"] = s["exact_match"]
    for d in rows.values():
        n = len(tok(d["response"], add_special_tokens=False).input_ids)
        d["n_tokens_retokenized"] = n
        # stop strings are removed by lm-eval, so a response close to the cap never stopped by itself
        d["truncated"] = n >= C.GSM8K_MAX_NEW - 8
        d["unextractable_flexible"] = d.get("extracted_flexible-extract") in ("[invalid]", "", None)
        d["unextractable_strict"] = d.get("extracted_strict-match") in ("[invalid]", "", None)
    return sorted(rows.values(), key=lambda r: r["doc_id"])


def ll_rows(samples):
    out = []
    for s in samples:
        r = dict(doc_id=s["doc_id"], target=s["target"])
        lls = [x[0] for x in s["filtered_resps"]] if s["filtered_resps"] and isinstance(s["filtered_resps"][0], (list, tuple)) else s["filtered_resps"]
        r["loglikelihoods"] = [float(x) for x in lls]
        for k in ("acc", "acc_norm"):
            if k in s:
                r[k] = s[k]
        out.append(r)
    return sorted(out, key=lambda r: r["doc_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=[c[0] for c in C.CONDITIONS])
    ap.add_argument("--gen_bs", type=int, default=32)
    ap.add_argument("--ll_bs", type=int, default=4,
                    help="32 needs ~24 GB: full-vocab log-probs for the longest BoolQ passages; 4 is meant to fit 8 GB")
    ap.add_argument("--limit", type=int, default=None, help="docs per task (debug only; full splits otherwise)")
    ap.add_argument("--skip", nargs="*", default=[], help="tasks to skip, e.g. gsm8k_cot")
    ap.add_argument("--smoke", action="store_true", help="evaluate the tiny *_smoke training runs")
    args = ap.parse_args()
    if args.smoke:
        import task1.modeling as M
        M.SMOKE = True

    name = args.condition
    out = C.task_dir() / "eval" / (name.replace("@", "_at_") + ("_limit" + str(args.limit) if args.limit else "")
                                   + ("_smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    summary = dict(condition=name, limit=args.limit, env=env_info(), scores={}, timing_s={})
    if (out / "summary.json").exists() and args.skip:  # partial re-run: keep the other tasks' results
        from common.utils import load_json
        summary.update({k: v for k, v in load_json(out / "summary.json").items() if k != "env"})

    method, update = condition_spec(name)
    if method is not None and C.METHODS[method]["kind"] != "full":
        summary["merge_check"] = merge_check(method, update, tok)
        print("merge check:", summary["merge_check"])

    model = load_condition(name)
    torch.cuda.reset_peak_memory_stats()
    # enable_thinking=False is forwarded to Qwen3's chat template (non-thinking mode)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.gen_bs, enable_thinking=False)

    if C.GSM8K_TASK not in args.skip:
        t0 = time.perf_counter()
        r = simple_evaluate(model=lm, tasks=[C.GSM8K_TASK], num_fewshot=0, apply_chat_template=True,
                            gen_kwargs=f"max_gen_toks={C.GSM8K_MAX_NEW}", log_samples=True,
                            limit=args.limit, **_seed_kwargs())
        summary["timing_s"][C.GSM8K_TASK] = time.perf_counter() - t0
        summary.setdefault("run_info", {})[C.GSM8K_TASK] = dict(batch_size=args.gen_bs, gpu=torch.cuda.get_device_name())
        summary["scores"][C.GSM8K_TASK] = {k: v for k, v in r["results"][C.GSM8K_TASK].items() if k != "alias"}
        summary["task_configs"] = {C.GSM8K_TASK: r["configs"][C.GSM8K_TASK]}
        rows = gsm8k_rows(r["samples"][C.GSM8K_TASK], tok)
        write_jsonl(rows, out / "gsm8k_samples.jsonl")
        n = len(rows)
        summary["gsm8k_diagnostics"] = dict(
            n=n, truncation_rate=sum(x["truncated"] for x in rows) / n,
            unextractable_flexible_rate=sum(x["unextractable_flexible"] for x in rows) / n,
            unextractable_strict_rate=sum(x["unextractable_strict"] for x in rows) / n,
            mean_response_tokens=sum(x["n_tokens_retokenized"] for x in rows) / n)
        summary["gsm8k_diagnostics"]["prompts_have_empty_think_block"] = all(
            x["prompt"] is None or "<think>\n\n</think>" in x["prompt"] for x in rows)
        print(summary["scores"][C.GSM8K_TASK], summary["gsm8k_diagnostics"])
        save_json(summary, out / "summary.json")

    tasks = [t for t in C.LL_TASKS if t not in args.skip]
    if tasks:
        lm.batch_size_per_gpu = args.ll_bs
        t0 = time.perf_counter()
        r = simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, apply_chat_template=False,
                            log_samples=True, limit=args.limit, **_seed_kwargs())
        summary["timing_s"]["loglikelihood_tasks"] = time.perf_counter() - t0
        summary.setdefault("run_info", {})["loglikelihood_tasks"] = dict(batch_size=args.ll_bs,
                                                                         gpu=torch.cuda.get_device_name())
        summary.setdefault("task_configs", {}).update({t: r["configs"][t] for t in tasks})
        for t in tasks:
            summary["scores"][t] = {k: v for k, v in r["results"][t].items() if k != "alias"}
            write_jsonl(ll_rows(r["samples"][t]), out / f"{t}_samples.jsonl")
            print(t, summary["scores"][t])

    summary["primary"] = {t: summary["scores"][t].get(C.PRIMARY_METRIC[t]) for t in summary["scores"]}
    summary["peak_gpu_mem_bytes"] = torch.cuda.max_memory_allocated()
    save_json(summary, out / "summary.json")
    print("primary:", summary["primary"])


if __name__ == "__main__":
    main()
