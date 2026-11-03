"""Offline selection experiments on the supplied pools (PA3 Task 3, steps 1-2 and the inputs to step 5).

    python -m task3.offline       # -> outputs/task3/offline/<ds>_{selections,contexts}.parquet

For every question, each of the 10 saved rollout orders and each budget N in {1,2,4,8,16,32}, the first N
rollouts of that order form the candidate set. Every method selects an answer group from that set only.
Rows record the selected rollout ID (for vote-based rules: the earliest rollout supporting the winner),
its answer group, correctness and token cost. The contexts table holds coverage, distinct answers and the
self-consistency / RWC statistics that the adaptive stopping simulation (step 5) reuses at n = 4, 8, 16, 32.
"""
import argparse
import time

import numpy as np
import pandas as pd

from task3 import config as C
from task3.data import Pool
from task3.selection import highest, rwc, visconf, vote

METHODS_BOTH = ["first", "sc", "logp_max", "logp_rwc", "selfcert_max", "selfcert_rwc", "shortest"]
METHODS_VLM = ["vision_max", "vision_rwc", "visconf_max", "visconf_rwc", "fusion_max", "fusion_rwc"]


def load_answers(ds):
    a = pd.read_parquet(C.task_dir() / "answers" / f"{ds}.parquet").sort_values(["q", "rollout_id"])
    G = a.group.to_numpy().reshape(-1, C.N_ROLLOUTS)
    Y = a.correct.to_numpy().reshape(-1, C.N_ROLLOUTS)
    return a, G, Y


def rollout_scores(pool):
    s = dict(T=pool.n_tokens.astype(np.float64), L=pool.rollout_means(pool.logp), K=pool.rollout_means(pool.kl))
    if pool.vis is not None:
        s["V"] = pool.rollout_means(pool.vis)
    return {k: v.reshape(-1, C.N_ROLLOUTS) for k, v in s.items()}


def run(ds):
    pool = Pool(ds)
    _, G, Y = load_answers(ds)
    S = rollout_scores(pool)
    vlm = pool.vis is not None
    qkey = "visconf" if vlm else "selfcert"
    sel_rows, ctx_rows = [], []
    t0 = time.time()
    for q in range(pool.n_q):
        if q % 100 == 0:
            print(f"  {ds}: question {q}/{pool.n_q} ({time.time() - t0:.0f}s)", flush=True)
        g, y = G[q], Y[q]
        T, L, K = S["T"][q], S["L"][q], S["K"][q]
        if vlm:
            rows = [pool.row(q, r) for r in range(C.N_ROLLOUTS)]
            kl_r = [pool.tokens(pool.kl, i) for i in rows]
            vis_r = [pool.tokens(pool.vis, i) for i in rows]
            V = S["V"][q]
        for o in range(C.N_ORDERS):
            order = pool.orders[q, o]
            for N in C.BUDGETS:
                cand = order[:N]
                gc, yc = g[cand], y[cand]
                cost_all, cost_first = float(T[cand].sum()), float(T[cand[0]])
                out = {}

                def put(name, pos=None, grp=None):
                    if grp is None:
                        grp = int(gc[pos])
                        rid = int(cand[pos])
                    else:
                        rid = int(cand[pos])
                    out[name] = (rid, grp)

                put("first", 0)
                sg, sf, uniq, W = vote(gc)
                put("sc", sf, sg)
                put("logp_max", highest(L[cand]))
                rg, rf, _ = rwc(gc, L[cand])
                put("logp_rwc", rf, rg)
                put("selfcert_max", highest(K[cand]))
                kg, kf, kmargin = rwc(gc, K[cand])
                put("selfcert_rwc", kf, kg)
                put("shortest", int(np.argmin(T[cand])))
                qrwc_g, qrwc_m = kg, kmargin
                if vlm:
                    seg = np.concatenate([[0], np.cumsum([len(kl_r[i]) for i in cand])])
                    Ci, Z, Zt, _ = visconf(np.concatenate([kl_r[i] for i in cand]),
                                           np.concatenate([vis_r[i] for i in cand]), seg)
                    put("vision_max", highest(V[cand]))
                    vg, vf, _ = rwc(gc, V[cand])
                    put("vision_rwc", vf, vg)
                    put("visconf_max", highest(Z))
                    zg, zf, zmargin = rwc(gc, Z)
                    put("visconf_rwc", zf, zg)
                    put("fusion_max", highest(Zt))
                    fg, ff, _ = rwc(gc, Zt)
                    put("fusion_rwc", ff, fg)
                    qrwc_g, qrwc_m = zg, zmargin
                corr_group = set(np.unique(gc[yc]).tolist())
                for m, (rid, grp) in out.items():
                    sel_rows.append((q, o, N, m, rid, grp, grp in corr_group and grp != 0,
                                     cost_first if m == "first" else cost_all))
                top = W.max()
                ctx_rows.append((q, o, N, bool(yc.any()), len(uniq), cost_all, cost_first,
                                 float(top / N), int((W == top).sum() == 1), sg, bool(sg in corr_group),
                                 qrwc_g, float(qrwc_m), float(yc.mean())))
    sel = pd.DataFrame(sel_rows, columns=["q", "order", "N", "method", "sel_rollout", "group", "correct", "cost"])
    ctx = pd.DataFrame(ctx_rows, columns=["q", "order", "N", "coverage", "n_distinct", "cost_all", "cost_first",
                                          "sc_top_frac", "sc_unique", "sc_group", "sc_correct",
                                          f"{qkey}_rwc_group", f"{qkey}_rwc_margin", "frac_correct_in_prefix"])
    for df in (sel, ctx):
        df["split"] = pool.split_of[df.q.to_numpy()]
    out = C.task_dir() / "offline"
    out.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(out / f"{ds}_selections.parquet")
    ctx.to_parquet(out / f"{ds}_contexts.parquet")
    print(f"{ds}: {len(sel)} selections, {len(ctx)} contexts in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    for d in ap.parse_args().datasets:
        run(d)
