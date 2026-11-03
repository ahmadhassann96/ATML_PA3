"""Adaptive sampling (PA3 Task 3, step 5): offline simulation of stopping between completed rollouts.

    python -m task3.adaptive      # -> outputs/task3/adaptive/

Rollouts are revealed in the saved orders. After n in {4, 8, 16} candidates a controller may stop;
otherwise it consumes all 32. Both controllers return the current self-consistency answer.
* agreement(tau): a unique, non-NO_ANSWER leading answer holds at least a fraction tau of the n votes.
* quality(tau, gamma): agreement(tau), the self-consistency and RWC winners agree, and the RWC vote margin
  (largest minus second-largest answer vote, over total vote weight) is at least gamma. RWC uses
  self-certainty for MATH and VisConf for MathVista, recomputed (including VisConf calibration and ranks)
  on the revealed candidates only.
Settings are chosen on the development split (lowest mean token cost whose accuracy is within 1 pp of
fixed-32 self-consistency; ties by higher accuracy, then smaller tau and gamma), frozen, then evaluated
on the held-out split. The always-32 policy is included as the fallback in each family.
"""
import argparse

import numpy as np
import pandas as pd

from common.utils import save_json
from task3 import config as C

STOPS = C.CHECKPOINTS + [32]


def _arrays(ds):
    ctx = pd.read_parquet(C.task_dir() / "offline" / f"{ds}_contexts.parquet")
    qkey = C.DATASETS[ds]["quality"]
    ctx = ctx[ctx.N.isin(STOPS)].sort_values(["q", "order", "N"])
    nq, no, ns = ctx.q.nunique(), C.N_ORDERS, len(STOPS)

    def A(col, dtype=float):
        return ctx[col].to_numpy().astype(dtype).reshape(nq, no, ns)

    return dict(split=ctx.drop_duplicates("q").sort_values("q").split.to_numpy(),
                top=A("sc_top_frac"), unique=A("sc_unique", bool), group=A("sc_group", int),
                correct=A("sc_correct", bool), cost=A("cost_all"),
                rwc_group=A(f"{qkey}_rwc_group", int), margin=A(f"{qkey}_rwc_margin"),
                frac_correct=A("frac_correct_in_prefix"))


def simulate(a, family, tau, gamma):
    """Stop index per (question, order), plus correctness and cost at the stop."""
    if family == "always32":
        cond = np.zeros_like(a["top"], dtype=bool)
    else:
        cond = a["unique"] & (a["group"] != 0) & (a["top"] >= tau - 1e-12)
        if family == "quality":
            cond &= (a["rwc_group"] == a["group"]) & (a["margin"] >= gamma - 1e-12)
    cond[..., -1] = True                                  # always stop at 32
    stop = cond.argmax(axis=-1)
    take = lambda x: np.take_along_axis(x, stop[..., None], axis=-1)[..., 0]  # noqa: E731
    return stop, take(a["correct"]), take(a["cost"])


def _summ(stop, correct, cost, mask):
    acc_q = correct[mask].mean(axis=1)                    # average over orders, then questions
    n = np.array(STOPS)[stop[mask]]
    return dict(acc=float(acc_q.mean()), cost=float(cost[mask].mean()), rollouts=float(n.mean()),
                stop_dist={str(s): float((n == s).mean()) for s in STOPS})


def grid():
    yield "always32", None, None
    for t in C.TAUS:
        yield "agreement", t, None
    for t in C.TAUS:
        for g in C.GAMMAS:
            yield "quality", t, g


def run(ds):
    a = _arrays(ds)
    dev, ev = a["split"] == "dev", a["split"] == "eval"
    sc32 = {k: a["correct"][m][..., -1].mean() for k, m in (("dev", dev), ("eval", ev))}
    rows = []
    for fam, t, g in grid():
        stop, cor, cost = simulate(a, fam, t, g)
        for split, m in (("dev", dev), ("eval", ev)):
            rows.append(dict(family=fam, tau=t, gamma=g, split=split, **_summ(stop, cor, cost, m)))
    res = pd.DataFrame(rows)
    out = C.task_dir() / "adaptive"
    out.mkdir(parents=True, exist_ok=True)
    res.drop(columns="stop_dist").to_csv(out / f"{ds}_grid.csv", index=False)

    chosen = {}
    for fam in ("agreement", "quality"):
        d = res[(res.split == "dev") & res.family.isin([fam, "always32"])].copy()
        d = d[d.acc >= sc32["dev"] - C.DEV_TOLERANCE - 1e-12]
        d["tau_k"], d["gamma_k"] = d.tau.fillna(np.inf), d.gamma.fillna(np.inf)
        best = d.sort_values(["cost", "acc", "tau_k", "gamma_k"], ascending=[True, False, True, True]).iloc[0]
        fam_b, t, g = best.family, (None if pd.isna(best.tau) else float(best.tau)), \
            (None if pd.isna(best.gamma) else float(best.gamma))
        stop, cor, cost = simulate(a, fam_b, t, g)
        e = _summ(stop, cor, cost, ev)
        # incorrect early stops, and those where fixed-32 self-consistency would have been right
        early = (stop < len(STOPS) - 1) & ev[:, None]
        wrong_early = early & ~cor
        recover = wrong_early & a["correct"][..., -1]
        full_frac = a["frac_correct"][..., -1]
        bins = pd.cut(full_frac[ev].ravel(), [-0.01, 0, 0.25, 0.5, 0.75, 0.99, 1.0],
                      labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"])
        by_frac = pd.DataFrame(dict(bin=bins, n=np.array(STOPS)[stop[ev]].ravel(), correct=cor[ev].ravel())) \
            .groupby("bin", observed=False).agg(mean_rollouts=("n", "mean"), acc=("correct", "mean"), count=("n", "size"))
        chosen[fam] = dict(selected=dict(policy=fam_b, tau=t, gamma=g),
                           dev=_summ(stop, cor, cost, dev), eval=e,
                           eval_sc32_acc=float(sc32["eval"]), dev_sc32_acc=float(sc32["dev"]),
                           eval_incorrect_early_stops=float(wrong_early.sum() / ev.sum() / C.N_ORDERS),
                           eval_early_stops=float(early.sum() / ev.sum() / C.N_ORDERS),
                           eval_incorrect_early_recoverable_by_sc32=float(recover.sum() / ev.sum() / C.N_ORDERS),
                           eval_by_full_pool_correct_fraction=by_frac.reset_index().to_dict("records"),
                           acc_diff_vs_sc32=_boot(cor[ev].mean(1), a["correct"][ev][..., -1].mean(1)))
    fixed = {}
    ctx = pd.read_parquet(C.task_dir() / "offline" / f"{ds}_contexts.parquet")
    for N, d in ctx[ctx.split == "eval"].groupby("N"):
        qd = d.groupby("q")[["sc_correct", "cost_all"]].mean()
        fixed[int(N)] = dict(acc=float(qd.sc_correct.mean()), cost=float(qd.cost_all.mean()))
    summary = dict(dataset=ds, quality_signal=C.DATASETS[ds]["quality"], checkpoints=C.CHECKPOINTS,
                   taus=C.TAUS, gammas=C.GAMMAS, dev_tolerance_pp=100 * C.DEV_TOLERANCE,
                   chosen=chosen, fixed_sc_eval=fixed)
    save_json(summary, out / f"{ds}_summary.json")
    for fam, c in chosen.items():
        print(ds, fam, c["selected"], "eval acc %.3f cost %.0f rollouts %.1f | SC32 %.3f, fixed cost %.0f"
              % (c["eval"]["acc"], c["eval"]["cost"], c["eval"]["rollouts"], c["eval_sc32_acc"], fixed[32]["cost"]))


def _boot(a, b, n=C.N_BOOT):
    rng = np.random.default_rng(C.SEED)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    for d in ap.parse_args().datasets:
        run(d)
