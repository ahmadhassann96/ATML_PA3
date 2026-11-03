"""Load and join the supplied rollout tables, verify token-list alignment, build splits and orders.

    python -m task3.data            # both datasets -> cache + outputs/task3/{splits,orders,data_checks}

Rollouts are stored in a fixed layout: question q (index into the lexicographically sorted IDs) and
rollout r occupy row q * 32 + r. Token-level lists are flattened into one float32 array per measurement
with offsets, so all per-token work is vectorized.
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.utils import load_json, save_json
from task3 import config as C

GEN_COLS = ["dataset", "model", "sample_id", "rollout_id", "question_text", "generated_text",
            "n_generated_text_tokens", "hit_max_new_tokens", "gold_answer", "extracted_answer"]


def _sorted_take(table):
    """Row order (sample_id lexicographic, rollout_id) for a pyarrow table."""
    idx = pc.sort_indices(table, sort_keys=[("sample_id", "ascending"), ("rollout_id", "ascending")])
    return table.take(idx)


def _flat(table, col):
    arr = table.column(col).combine_chunks()
    offs = np.asarray(arr.offsets)
    vals = np.asarray(arr.flatten(), dtype=np.float32) if col != "generated_token_ids" else None
    return vals, offs


def build(ds):
    spec = C.DATASETS[ds]
    folder = C.DATA_ROOT / spec["folder"]
    checks = {}

    gen = _sorted_take(pq.read_table(folder / "generations.parquet"))
    prob = _sorted_take(pq.read_table(folder / "probability_metrics.parquet"))
    tables = {"generations": gen, "probability_metrics": prob}
    if spec["vlm"]:
        tables["attention_metrics"] = _sorted_take(pq.read_table(folder / "attention_metrics.parquet"))

    # keys must match row by row across files, 32 rollouts per question
    keys = {name: (t.column("dataset").to_pylist(), t.column("sample_id").to_pylist(),
                   t.column("rollout_id").to_pylist()) for name, t in tables.items()}
    ref = keys["generations"]
    for name, k in keys.items():
        assert k == ref, f"{ds}: join keys differ between generations and {name}"
    ids = sorted(set(ref[1]))
    n_q = len(ids)
    assert len(ref[1]) == n_q * C.N_ROLLOUTS and ref[2] == list(range(C.N_ROLLOUTS)) * n_q, "rollout layout"
    assert ref[1] == [i for i in ids for _ in range(C.N_ROLLOUTS)], "question order"
    checks.update(n_questions=n_q, n_rollouts=len(ref[1]), datasets=sorted(set(ref[0])))

    # token-list alignment
    n_tok = np.asarray(gen.column("n_generated_text_tokens").to_numpy(), dtype=np.int64)
    tok_offs = np.asarray(gen.column("generated_token_ids").combine_chunks().offsets)
    logp, offs = _flat(prob, "logp")
    kl, offs_kl = _flat(prob, "kl_u_p")
    assert np.array_equal(np.diff(tok_offs), n_tok), "token ids vs n_generated_text_tokens"
    assert np.array_equal(offs, tok_offs) and np.array_equal(offs_kl, tok_offs), "logp/kl vs token ids"
    arrays = dict(offsets=offs, logp=logp, kl=kl, n_tokens=n_tok)
    checks["token_lists_aligned"] = True
    checks["total_tokens"] = int(n_tok.sum())
    checks["zero_length_rollouts"] = int((n_tok == 0).sum())
    checks["nan_logp"] = int(np.isnan(logp).sum())
    checks["nan_kl"] = int(np.isnan(kl).sum())

    if spec["vlm"]:
        att = tables["attention_metrics"]
        img, o1 = _flat(att, "img_attn_total")
        prm, o2 = _flat(att, "prompt_text_attn_total")
        gtx, o3 = _flat(att, "generated_text_attn_total")
        cnt, o4 = _flat(att, "n_generated_text_tokens")
        for o in (o1, o2, o3, o4):
            assert np.array_equal(o, offs), "attention lists vs token ids"
        # preceding-token counts must read 0..T-1 inside every rollout
        expect = np.arange(len(cnt)) - np.repeat(offs[:-1], np.diff(offs))
        assert np.array_equal(cnt.astype(np.int64), expect), "attention n_generated_text_tokens is not 0..T-1"
        den = img + prm + gtx
        checks["attn_zero_denominator"] = int((den <= 0).sum())
        arrays["vis"] = np.where(den > 0, img / np.maximum(den, 1e-30), 0.0).astype(np.float32)
        arrays["img_attn"] = img
        checks["attention_aligned"] = True
        checks["first_token_generated_attn_zero"] = bool(np.all(gtx[offs[:-1][n_tok > 0]] == 0))

    meta = gen.select(GEN_COLS).to_pandas()
    meta["q"] = np.repeat(np.arange(n_q), C.N_ROLLOUTS)
    checks["capped_rollouts"] = int(meta.hit_max_new_tokens.sum())
    checks["null_extractions"] = int(meta.extracted_answer.isna().sum())

    cache = C.cache_dir()
    np.savez(cache / f"{ds}_tokens.npz", **arrays)
    meta.to_parquet(cache / f"{ds}_rollouts.parquet")

    # development / evaluation split: sorted IDs permuted with default_rng(6304)
    perm = np.random.default_rng(C.SEED).permutation(np.array(ids, dtype=object))
    split = dict(seed=C.SEED, rule="sorted IDs permuted with numpy default_rng(6304); first 200 dev",
                 dev=list(perm[:C.N_DEV]), eval=list(perm[C.N_DEV:]))
    if ds == "math":
        split["live50"] = split["eval"][:C.LIVE_N_QUESTIONS]
    save_json(split, C.task_dir() / "splits" / f"{ds}.json")

    # ten seeded permutations of the 32 rollout IDs per question (questions in sorted-ID order)
    rng = np.random.default_rng(C.SEED)
    orders = np.stack([np.stack([rng.permutation(C.N_ROLLOUTS) for _ in range(C.N_ORDERS)]) for _ in ids])
    (C.task_dir() / "orders").mkdir(parents=True, exist_ok=True)
    np.save(C.task_dir() / "orders" / f"{ds}.npy", orders.astype(np.int8))
    save_json(dict(ids=ids, shape=list(orders.shape), seed=C.SEED,
                   rule="one default_rng(6304); for each question in sorted-ID order draw 10 permutations"),
              C.task_dir() / "orders" / f"{ds}_meta.json")

    save_json(checks, C.task_dir() / "data_checks" / f"{ds}.json")
    print(ds, json.dumps(checks))


class Pool:
    """All rollouts of one dataset, in the fixed (question, rollout) layout."""

    def __init__(self, ds):
        self.ds = ds
        self.spec = C.DATASETS[ds]
        cache = C.cache_dir()
        if not (cache / f"{ds}_tokens.npz").exists():
            build(ds)
        z = np.load(cache / f"{ds}_tokens.npz")
        self.offsets = z["offsets"]
        self.logp, self.kl, self.n_tokens = z["logp"], z["kl"], z["n_tokens"]
        self.vis = z["vis"] if "vis" in z else None
        self.meta = pd.read_parquet(cache / f"{ds}_rollouts.parquet")
        self.ids = sorted(self.meta.sample_id.unique())
        self.n_q = len(self.ids)
        self.orders = np.load(C.task_dir() / "orders" / f"{ds}.npy").astype(np.int64)
        split = load_json(C.task_dir() / "splits" / f"{ds}.json")
        pos = {s: i for i, s in enumerate(self.ids)}
        self.split = {k: np.array([pos[s] for s in v]) for k, v in split.items() if isinstance(v, list)}
        self.split_of = np.empty(self.n_q, dtype=object)
        self.split_of[self.split["dev"]] = "dev"
        self.split_of[self.split["eval"]] = "eval"

    def row(self, q, r):
        return q * C.N_ROLLOUTS + r

    def tokens(self, arr, row):
        return arr[self.offsets[row]:self.offsets[row + 1]]

    def rollout_means(self, arr):
        """Mean of a per-token array within every rollout (0 for empty rollouts)."""
        padded = np.append(arr.astype(np.float64), 0.0)   # keeps reduceat valid if the last rollout is empty
        sums = np.add.reduceat(padded, self.offsets[:-1])
        sums[self.n_tokens == 0] = 0.0                    # reduceat returns arr[i] for empty segments
        return np.where(self.n_tokens > 0, sums / np.maximum(self.n_tokens, 1), 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    for ds in ap.parse_args().datasets:
        build(ds)
