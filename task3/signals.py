"""What the quality signals capture (PA3 Task 3, step 3), using each question's full pool of 32 rollouts.

    python -m task3.signals      # -> outputs/task3/signals/

* Within-question associations between length, scores and correctness: Spearman correlations and the
  top-minus-bottom quartile correctness difference (8 highest vs 8 lowest rollouts per score, ties broken
  by rollout ID), averaged across questions.
* MathVista: within-rollout correlation between calibrated confidence c_t and visual engagement v_t,
  summarized across questions; check of Z^token - Z = Cov_t(c, v); illustrative trajectories chosen by a
  fixed rule (first evaluation-split questions, in split order, where VisConf+RWC and self-consistency
  pick different answers at N = 32 in order 0 and exactly one of them is correct).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common.utils import save_json  # noqa: E402
from task3 import config as C  # noqa: E402
from task3.data import Pool  # noqa: E402
from task3.offline import load_answers, rollout_scores  # noqa: E402
from task3.selection import visconf  # noqa: E402

LABEL = {"T": "length (tokens)", "L": "mean log-prob", "K": "self-certainty", "V": "visual engagement",
         "Z": "VisConf", "Zt": "tokenwise fusion"}


def _quartile_diff(s, y):
    order = np.lexsort((np.arange(len(s)), -s))   # descending score, ties by rollout ID
    k = len(s) // 4
    return y[order[:k]].mean() - y[order[-k:]].mean()


def _spear(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return spearmanr(a, b).statistic


def run(ds):
    pool = Pool(ds)
    _, G, Y = load_answers(ds)
    S = rollout_scores(pool)
    vlm = pool.vis is not None
    out = C.task_dir() / "signals"
    out.mkdir(parents=True, exist_ok=True)
    keys = ["T", "L", "K"] + (["V", "Z", "Zt"] if vlm else [])
    rows, cv_rows, fusion_err = [], [], 0.0
    for q in range(pool.n_q):
        y = Y[q].astype(float)
        sc = {k: S[k][q] for k in ("T", "L", "K") + (("V",) if vlm else ())}
        if vlm:
            idx = [pool.row(q, r) for r in range(C.N_ROLLOUTS)]
            kl = [pool.tokens(pool.kl, i) for i in idx]
            vs = [pool.tokens(pool.vis, i) for i in idx]
            seg = np.concatenate([[0], np.cumsum([len(x) for x in kl])])
            Ci, Z, Zt, c = visconf(np.concatenate(kl), np.concatenate(vs), seg)
            sc["Z"], sc["Zt"] = Z, Zt
            for r in range(C.N_ROLLOUTS):
                cr, vr = c[seg[r]:seg[r + 1]], vs[r]
                cov = float(np.mean(cr * vr) - cr.mean() * vr.mean()) if len(cr) else 0.0
                fusion_err = max(fusion_err, abs((Zt[r] - Z[r]) - cov))
                rho = (np.corrcoef(cr, vr)[0, 1] if len(cr) > 2 and np.ptp(cr) > 0 and np.ptp(vr) > 0 else np.nan)
                cv_rows.append(dict(q=q, rollout_id=r, split=pool.split_of[q], correct=bool(Y[q, r]),
                                    corr_c_v=rho, cov_c_v=cov, C=Ci[r], V=sc["V"][r], Z=Z[r], Zt=Zt[r]))
        row = dict(q=q, split=pool.split_of[q], frac_correct=y.mean(), mixed=bool(0 < y.mean() < 1))
        for k in keys:
            row[f"quartile_diff_{k}"] = _quartile_diff(sc[k], y)
            row[f"spearman_{k}_correct"] = _spear(sc[k], y)
            if k != "T":
                row[f"spearman_T_{k}"] = _spear(sc["T"], sc[k])
        rows.append(row)
    per_q = pd.DataFrame(rows)
    per_q.to_parquet(out / f"{ds}_per_question.parquet")

    summ = []
    for split in ("eval", "dev"):
        d = per_q[per_q.split == split]
        m = d[d.mixed]
        for k in keys:
            summ.append(dict(dataset=ds, split=split, signal=k, label=LABEL[k],
                             quartile_diff_all=d[f"quartile_diff_{k}"].mean(),
                             quartile_diff_mixed=m[f"quartile_diff_{k}"].mean(),
                             spearman_with_correct_mixed=m[f"spearman_{k}_correct"].mean(),
                             spearman_with_length=(d[f"spearman_T_{k}"].mean() if k != "T" else 1.0),
                             n_questions=len(d), n_mixed=len(m)))
    pd.DataFrame(summ).to_csv(out / f"{ds}_associations.csv", index=False)
    print(pd.DataFrame(summ)[lambda x: x.split == "eval"].round(3).to_string(index=False))

    if vlm:
        cv = pd.DataFrame(cv_rows)
        cv.to_parquet(out / f"{ds}_conf_vision_rollouts.parquet")
        e = cv[cv.split == "eval"]
        per_question = e.groupby("q").corr_c_v.mean()
        res = dict(fusion_identity_max_abs_error=fusion_err,
                   within_rollout_corr_c_v=dict(
                       question_mean_of_rollout_means=float(per_question.mean()),
                       question_median=float(per_question.median()),
                       question_q25=float(per_question.quantile(0.25)), question_q75=float(per_question.quantile(0.75)),
                       frac_questions_negative=float((per_question < 0).mean()),
                       correct_rollouts_mean=float(e[e.correct].corr_c_v.mean()),
                       incorrect_rollouts_mean=float(e[~e.correct].corr_c_v.mean())),
                   mean_cov_c_v=float(e.cov_c_v.mean()), split="eval")
        save_json(res, out / f"{ds}_conf_vision.json")
        print(res)
        trajectories(pool, G, Y, sc_unused=None)


def trajectories(pool, G, Y, sc_unused, n_examples=3, smooth=15):
    """Fixed-rule illustrative trajectories for MathVista (see module docstring)."""
    sel = pd.read_parquet(C.task_dir() / "offline" / "mathvista_selections.parquet")
    s = sel[(sel.N == 32) & (sel.order == 0)].pivot(index="q", columns="method", values=["group", "correct", "sel_rollout"])
    chosen = []
    for q in pool.split["eval"]:
        if s.loc[q, ("group", "visconf_rwc")] != s.loc[q, ("group", "sc")] and \
                s.loc[q, ("correct", "visconf_rwc")] != s.loc[q, ("correct", "sc")]:
            chosen.append(int(q))
        if len(chosen) == n_examples:
            break
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 2.6 * len(chosen)), squeeze=False)
    info = []
    for row, q in enumerate(chosen):
        idx = [pool.row(q, r) for r in range(C.N_ROLLOUTS)]
        kl = [pool.tokens(pool.kl, i) for i in idx]
        vs = [pool.tokens(pool.vis, i) for i in idx]
        seg = np.concatenate([[0], np.cumsum([len(x) for x in kl])])
        _, Z, _, c = visconf(np.concatenate(kl), np.concatenate(vs), seg)
        picks = {"VisConf+RWC pick": int(s.loc[q, ("sel_rollout", "visconf_rwc")]),
                 "self-consistency pick": int(s.loc[q, ("sel_rollout", "sc")])}
        for col, (name, r) in enumerate(picks.items()):
            ax = axes[row, col]
            cr, vr = c[seg[r]:seg[r + 1]], vs[r]
            k = np.ones(smooth) / smooth
            ax.plot(np.convolve(cr, k, mode="valid"), label="calibrated confidence $c_t$", lw=1)
            ax2 = ax.twinx()
            ax2.plot(np.convolve(vr, k, mode="valid"), color="#d62728", label="visual engagement $v_t$", lw=1)
            ax.set_title(f"q={pool.ids[q]}  {name}: rollout {r}, {'correct' if Y[q, r] else 'wrong'}, "
                         f"Z={Z[r]:.3f}", fontsize=8)
            ax.set_xlabel("generated token (moving average, 15)")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7)
                ax2.legend(loc="upper right", fontsize=7)
            info.append(dict(question=pool.ids[q], method=name, rollout=r, correct=bool(Y[q, r]),
                             answer_group=int(G[q, r]), Z=float(Z[r]), n_tokens=int(len(cr))))
    fig.tight_layout()
    fig.savefig(C.task_dir() / "signals" / "mathvista_trajectories.png", dpi=140)
    plt.close(fig)
    rule = ("first evaluation-split questions (split order) where VisConf+RWC and self-consistency select different "
            "answers at N=32 in order 0 and exactly one of them is correct; both selected rollouts are shown")
    save_json(dict(rule=rule, examples=info), C.task_dir() / "signals" / "mathvista_trajectories.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    for d in ap.parse_args().datasets:
        run(d)
