"""Task 3 tables and figures (steps 1, 2 and 6) from the offline selection results.

    python -m task3.report      # -> outputs/task3/report/

Per-question outcomes are averaged over the 10 orders first, then over questions. Headline numbers use
the 800-question evaluation split; development-split tables are written alongside. Bootstrap intervals
resample questions (paired, 1,000 resamples, seed 6304).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.utils import load_json, save_json  # noqa: E402
from task3 import config as C  # noqa: E402

NAMES = {"first": "FirstAns", "sc": "Self-consistency", "logp_max": "Max log-prob", "logp_rwc": "Log-prob + RWC",
         "selfcert_max": "Max self-certainty", "selfcert_rwc": "Self-certainty + RWC", "shortest": "Shortest",
         "vision_max": "Max vision", "vision_rwc": "Vision + RWC", "visconf_max": "Max VisConf",
         "visconf_rwc": "VisConf + RWC", "fusion_max": "Max tokenwise fusion", "fusion_rwc": "Tokenwise fusion + RWC"}
MAIN = ["first", "sc", "logp_max", "selfcert_max", "selfcert_rwc"]
MAIN_VLM = MAIN + ["vision_max", "vision_rwc", "visconf_max", "visconf_rwc"]
COMPARISONS = [("sc", "first"), ("selfcert_max", "sc"), ("selfcert_rwc", "sc"), ("selfcert_rwc", "selfcert_max"),
               ("logp_max", "sc"), ("shortest", "sc")]
COMPARISONS_VLM = [("visconf_rwc", "sc"), ("visconf_rwc", "visconf_max"), ("visconf_rwc", "selfcert_rwc"),
                   ("visconf_rwc", "vision_rwc"), ("vision_rwc", "sc"), ("fusion_rwc", "visconf_rwc"),
                   ("fusion_max", "visconf_max")]


def out_dir():
    d = C.task_dir() / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def per_question(ds):
    sel = pd.read_parquet(C.task_dir() / "offline" / f"{ds}_selections.parquet")
    ctx = pd.read_parquet(C.task_dir() / "offline" / f"{ds}_contexts.parquet")
    pq_sel = sel.groupby(["split", "q", "N", "method"], as_index=False)[["correct", "cost"]].mean()
    pq_ctx = ctx.groupby(["split", "q", "N"], as_index=False)[["coverage", "n_distinct", "cost_all"]].mean()
    return pq_sel, pq_ctx


def boot(diff, n=C.N_BOOT):
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    b = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def tables(ds, pq_sel, pq_ctx):
    res = {}
    for split in ("eval", "dev"):
        s = pq_sel[pq_sel.split == split]
        c = pq_ctx[pq_ctx.split == split]
        acc = s.groupby(["method", "N"]).agg(acc=("correct", "mean"), cost=("cost", "mean")).reset_index()
        cov = c.groupby("N").agg(coverage=("coverage", "mean"), distinct=("n_distinct", "mean")).reset_index()
        t = acc.merge(cov, on="N")
        t["selection_gap"] = t.coverage - t.acc
        t["dataset"], t["split"] = ds, split
        t["method_name"] = t.method.map(NAMES)
        t.to_csv(out_dir() / f"{ds}_{split}_methods_by_N.csv", index=False)
        res[split] = t
    # failure decomposition: no correct candidate generated vs available but missed
    s = pq_sel[pq_sel.split == "eval"].merge(pq_ctx[pq_ctx.split == "eval"], on=["split", "q", "N"])
    dec = s.groupby(["method", "N"]).apply(lambda d: pd.Series(dict(
        correct=d.correct.mean(), no_correct_generated=(1 - d.coverage).mean(),
        correct_available_but_missed=(d.coverage - d.correct).mean())), include_groups=False).reset_index()
    dec["dataset"] = ds
    dec.to_csv(out_dir() / f"{ds}_eval_failure_decomposition.csv", index=False)
    # paired bootstrap of the main accuracy differences, every budget
    comps = COMPARISONS + (COMPARISONS_VLM if ds == "mathvista" else [])
    e = pq_sel[pq_sel.split == "eval"].pivot_table(index=["N", "q"], columns="method", values="correct")
    rows = []
    for N in C.BUDGETS:
        x = e.loc[N]
        for a, b in comps:
            m, lo, hi = boot((x[a] - x[b]).to_numpy())
            rows.append(dict(dataset=ds, N=N, a=NAMES[a], b=NAMES[b], diff_pp=100 * m, ci_lo_pp=100 * lo,
                             ci_hi_pp=100 * hi))
    pd.DataFrame(rows).to_csv(out_dir() / f"{ds}_eval_bootstrap_differences.csv", index=False)
    return res


def figures(results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, ds in enumerate(C.DATASETS):
        t = results[ds]["eval"]
        methods = MAIN_VLM if ds == "mathvista" else MAIN + ["shortest"]
        ax = axes[row, 0]
        cov = t.drop_duplicates("N").sort_values("N")
        ax.plot(cov.N, 100 * cov.coverage, "k--", lw=2, label="Oracle coverage")
        for m in methods:
            d = t[t.method == m].sort_values("N")
            ax.plot(d.N, 100 * d.acc, marker="o", ms=3, label=NAMES[m])
        ax.set(xscale="log", xticks=C.BUDGETS, xticklabels=C.BUDGETS, xlabel="candidates N", ylabel="accuracy (%)",
               title=f"{C.DATASETS[ds]['label']}: accuracy vs N (eval)")
        ax.legend(fontsize=7)
        ax = axes[row, 1]
        for m in ["first", "sc", "selfcert_rwc"] + (["visconf_rwc"] if ds == "mathvista" else []):
            d = t[t.method == m].sort_values("N")
            ax.plot(d.cost / 1e3, 100 * d.acc, marker="o", ms=3, label=NAMES[m])
        all_cost = t[t.method == "sc"].sort_values("N")   # cost of consuming all N candidates
        ax.plot(all_cost.cost / 1e3, 100 * all_cost.coverage, "k--", label="Oracle coverage (all N candidates)")
        ax.set(xscale="log", xlabel="generated tokens per question (thousands)", ylabel="accuracy (%)",
               title="accuracy vs token cost")
        ax.legend(fontsize=7)
        ax = axes[row, 2]
        ax.plot(cov.N, cov.distinct, "o-", label="distinct answer groups")
        ax.set(xscale="log", xticks=C.BUDGETS, xticklabels=C.BUDGETS, xlabel="candidates N", ylabel="mean distinct answers")
        ax2 = ax.twinx()
        sc = t[t.method == "sc"].sort_values("N")
        ax2.plot(sc.N, 100 * sc.selection_gap, "s-", color="#d62728", label="coverage - SC accuracy")
        ax2.set_ylabel("selection gap (pp)")
        ax.set_title("answer diversity and selection gap")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir() / "selection_curves.png", dpi=140)
    plt.close(fig)

    # adaptive vs fixed budgets
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ds in zip(axes, C.DATASETS):
        p = C.task_dir() / "adaptive" / f"{ds}_summary.json"
        if not p.exists():
            continue
        a = load_json(p)
        fx = a["fixed_sc_eval"]
        ns = sorted(fx, key=int)
        ax.plot([fx[n]["cost"] / 1e3 for n in ns], [100 * fx[n]["acc"] for n in ns], "o-", label="fixed-N self-consistency")
        for n in ns:
            ax.annotate(f"N={n}", (fx[n]["cost"] / 1e3, 100 * fx[n]["acc"]), fontsize=7)
        for fam, mk in (("agreement", "^"), ("quality", "s")):
            c = a["chosen"][fam]
            ax.plot(c["eval"]["cost"] / 1e3, 100 * c["eval"]["acc"], mk, ms=9,
                    label=f"{fam}: {c['selected']['policy']} tau={c['selected']['tau']} gamma={c['selected']['gamma']}")
        g = pd.read_csv(C.task_dir() / "adaptive" / f"{ds}_grid.csv")
        g = g[(g.split == "eval") & (g.family != "always32")]
        ax.scatter(g.cost / 1e3, 100 * g.acc, s=8, c="grey", alpha=0.5, label="all grid settings (eval)")
        ax.set(xscale="log", xlabel="generated tokens per question (thousands)", ylabel="accuracy (%)",
               title=f"{C.DATASETS[ds]['label']}: adaptive stopping (eval)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir() / "adaptive_vs_fixed.png", dpi=140)
    plt.close(fig)


def novelty(ds):
    """Does a newly revealed rollout open a new answer, a new correct answer, or repeat one? (eval split)"""
    ans = pd.read_parquet(C.task_dir() / "answers" / f"{ds}.parquet").sort_values(["q", "rollout_id"])
    G = ans.group.to_numpy().reshape(-1, C.N_ROLLOUTS)
    Y = ans.correct.to_numpy().reshape(-1, C.N_ROLLOUTS)
    orders = np.load(C.task_dir() / "orders" / f"{ds}.npy").astype(int)
    split = pd.read_parquet(C.task_dir() / "offline" / f"{ds}_contexts.parquet").drop_duplicates("q") \
        .sort_values("q").split.to_numpy()
    rows = []
    for q in np.flatnonzero(split == "eval"):
        for o in range(C.N_ORDERS):
            seen, got = set(), False
            for k, r in enumerate(orders[q, o]):
                new = G[q, r] not in seen
                rows.append(dict(k=k + 1, new_answer=new, first_correct=bool(Y[q, r] and not got)))
                seen.add(G[q, r])
                got |= bool(Y[q, r])
    d = pd.DataFrame(rows).groupby("k").mean().reset_index()
    d.to_csv(out_dir() / f"{ds}_eval_novelty_by_position.csv", index=False)
    return d


def main():
    results, summary = {}, {}
    for ds in C.DATASETS:
        pq_sel, pq_ctx = per_question(ds)
        results[ds] = tables(ds, pq_sel, pq_ctx)
        novelty(ds)
        t = results[ds]["eval"]
        summary[ds] = {m: {int(n): dict(acc=float(r.acc), cost=float(r.cost)) for n, r in d.set_index("N").iterrows()}
                       for m, d in t.groupby("method")}
        summary[ds]["coverage"] = {int(r.N): float(r.coverage) for r in t.drop_duplicates("N").itertuples()}
        compact = t[t.N.isin([1, 4, 32])].pivot_table(index="method_name", columns="N", values="acc")
        print(f"\n{ds} (eval, accuracy %):\n", (100 * compact).round(1).to_string())
    save_json(summary, out_dir() / "summary.json")
    figures(results)
    print("wrote", out_dir())


if __name__ == "__main__":
    main()
