"""Final evaluation of a trained stage on both final-test suites under four image conditions.

    python -m task2.evaluate                      # runs/main, stage3 -> runs/main/eval/stage3/
    python -m task2.evaluate --run pool49
    python -m task2.evaluate --smoke

Greedy decoding, thinking disabled, at most 16 new tokens, batch size 8. Question and original references
are fixed across conditions: correct (original image), complementary (partner image of the pair),
mismatched (an image from a different pair of the same suite, fixed mapping shared by all runs) and
no_image (visual tokens and their separator removed; positions and masks rebuilt). Teacher-forced NLL
of the original reference (termination token excluded) is computed for every condition.
"""
import argparse
import time

import numpy as np
import torch

from common.utils import load_json, read_jsonl, save_json, write_jsonl
from task2 import config as C
from task2 import model as M
from task2.features import FeatureStore
from task2.train import batched_losses, pair_items
from task2.vqa_norm import normalize, soft_accuracy

SUITES = {"grounding": "ground_test", "vqa": "vqa_test"}


def mismatch_map():
    """Each question gets an image from another pair of its suite (never either image of its own pair).
    Built once with seed 6304 and saved, so every run and comparison uses the same mapping."""
    path = C.data_dir() / "mismatch_map.json"
    if path.exists():
        return load_json(path)
    out = {}
    for si, (suite, f) in enumerate(SUITES.items()):
        pairs = read_jsonl(C.data_dir() / f"{f}.jsonl")
        g = np.random.default_rng([C.SEED, 3000 + si])
        for pi, p in enumerate(pairs):
            for it in p["items"]:
                q = int(g.integers(len(pairs) - 1))
                q += q >= pi  # uniform over the other pairs
                other = pairs[q]["items"][int(g.integers(2))]
                out[str(it["qid"])] = [other["split"], other["image_id"]]
    save_json(out, path)
    return out


def score(suite, pred_norm, item):
    return soft_accuracy(pred_norm, item["refs"]) if suite == "vqa" else float(pred_norm == item["canonical"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--stage", default="stage3")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run = a.run + ("_smoke" if a.smoke else "")
    rd = C.run_dir(run)
    out = rd / "eval" / a.stage
    torch.cuda.reset_peak_memory_stats()
    tok, lm, conn, st = M.load_final(rd, a.stage)
    store = FeatureStore(st["pool"])
    prompter = M.Prompter(tok, store.n_tokens)
    stop = {prompter.im_end, tok.eos_token_id}
    mm = mismatch_map()
    rows, timing = [], {}
    for suite, f in SUITES.items():
        pairs = read_jsonl(C.data_dir() / f"{f}.jsonl")
        if a.smoke:
            pairs = pairs[:4]
        items = {it["qid"]: it for p in pairs for it in p["items"]}
        partner = {p["items"][j]["qid"]: p["items"][1 - j] for p in pairs for j in range(2)}
        ex = pair_items(pairs)
        for e, (p, j) in zip(ex, [(p, j) for p in pairs for j in range(2)]):
            e["qid"] = p["items"][j]["qid"]
            e["mismatched"] = tuple(mm[str(e["qid"])])
        image_of = {"correct": lambda e: (e["split"], e["image_id"]), "complementary": lambda e: e["partner"],
                    "mismatched": lambda e: e["mismatched"], "no_image": lambda e: None}
        for cond in C.CONDITIONS:
            fn = image_of[cond]
            nll = batched_losses(lm, conn, prompter, store, ex, fn, drop_termination=True, bs=C.GEN_BS)
            gen_time = 0.0
            preds = []
            for s in range(0, len(ex), C.GEN_BS):
                chunk = ex[s:s + C.GEN_BS]
                ims = [fn(e) for e in chunk]
                encs = [prompter.encode(e["question"], im is not None, None, "answer", True) for e, im in zip(chunk, ims)]
                feats = [store.get(*im) if im is not None else None for im in ims]
                b = M.to_device(M.collate(encs, feats, prompter.placeholder, left=True))
                t0 = M.now()
                with torch.autocast("cuda", dtype=torch.float16):
                    toks = M.greedy(lm, conn, b, C.GEN_MAX_NEW, stop)
                gen_time += M.now() - t0
                preds += [tok.decode(t, skip_special_tokens=True) for t in toks]
            timing[f"{suite}/{cond}"] = dict(total_s=gen_time, per_answer_s=gen_time / len(ex))
            for e, pr, l in zip(ex, preds, nll):
                it, pa = items[e["qid"]], partner[e["qid"]]
                pn = normalize(pr)
                rows.append(dict(suite=suite, condition=cond, pair_id=e["pair_id"], kind=e["qtype"], qid=e["qid"],
                                 question=e["question"], image=list(fn(e)) if fn(e) else None,
                                 prediction=pr, prediction_norm=pn, target=it["target"], canonical=it["canonical"],
                                 refs=it["refs"], partner_canonical=pa["canonical"], score=score(suite, pn, it),
                                 matches_canonical=pn == it["canonical"], follows_partner=pn == pa["canonical"],
                                 empty_answer=pn == "", ref_nll=l,
                                 gen_time_per_answer_s=gen_time / len(ex)))
            acc = np.mean([r["score"] for r in rows if r["suite"] == suite and r["condition"] == cond])
            print(f"{suite:9s} {cond:13s} acc {100 * acc:5.1f}  nll {np.mean(nll):.3f}  "
                  f"{gen_time / len(ex) * 1000:.0f} ms/answer", flush=True)
    write_jsonl(rows, out / "predictions.jsonl")
    save_json(dict(run=run, stage=a.stage, n_visual_tokens=store.n_tokens, timing=timing,
                   peak_gpu_mem_allocated_bytes=torch.cuda.max_memory_allocated(), gen_max_new=C.GEN_MAX_NEW,
                   gen_bs=C.GEN_BS, decoding="greedy", gpu=torch.cuda.get_device_name(0),
                   time=time.strftime("%Y-%m-%d %H:%M:%S")), out / "eval_info.json")
    print("wrote", out)


if __name__ == "__main__":
    main()
