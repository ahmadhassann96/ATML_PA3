"""Task 2 tables and figures from the training logs and final-test predictions.

    python -m task2.analyze                 # every run under outputs/task2/runs/ -> outputs/task2/report/
    python -m task2.analyze --labels_only   # just the label-inspection sheet (needs the images)

Pair bootstrap: 4,000 resamples of complete pairs (seed 6304); the same resampled pair indices are used for
every condition of a suite, so differences between conditions are paired.
"""
import argparse
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.utils import load_json, read_jsonl, save_json  # noqa: E402
from task2 import config as C  # noqa: E402

COND_NAMES = {"correct": "Correct", "complementary": "Complementary swap", "mismatched": "Mismatched",
              "no_image": "No image"}


def out_dir():
    d = C.task_dir() / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- final-test tables
def pair_table(df):
    """One row per (suite, condition, pair): mean score, both-correct, identical answers, mean NLL."""
    g = df.groupby(["suite", "condition", "pair_id", "kind"], sort=False)
    t = g.agg(score=("score", "mean"), nll=("ref_nll", "mean"), both=("matches_canonical", "all"),
              empty_answer=("empty_answer", "mean"), follows=("follows_partner", "mean"),
              keeps=("matches_canonical", "mean")).reset_index()
    ident = g.prediction_norm.agg(lambda s: s.iloc[0] == s.iloc[1]).rename("identical").reset_index()
    return t.merge(ident, on=["suite", "condition", "pair_id", "kind"])


def boot_ci(x, idx):
    b = x[idx].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def final_tables(run, df, info):
    pt = pair_table(df)
    rows, adv, swap = [], [], []
    for suite in ("grounding", "vqa"):
        p = pt[pt.suite == suite]
        pairs = p[p.condition == "correct"].pair_id.tolist()
        P = len(pairs)
        idx = np.random.default_rng([C.SEED, 4000]).integers(0, P, size=(C.N_BOOT, P))
        arr = {c: p[p.condition == c].set_index("pair_id").loc[pairs] for c in C.CONDITIONS}
        for c in C.CONDITIONS:
            d = df[(df.suite == suite) & (df.condition == c)]
            a = arr[c]
            lo, hi = boot_ci(a.score.to_numpy(), idx)
            r = dict(run=run, suite=suite, condition=c, n_questions=len(d), n_pairs=P, accuracy=d.score.mean(),
                     acc_ci_lo=lo, acc_ci_hi=hi, pair_accuracy=a.both.mean(), identical_answer_rate=a.identical.mean(),
                     empty_rate=d.empty_answer.mean(), ref_nll=d.ref_nll.mean(),
                     follows_partner_rate=d.follows_partner.mean(),
                     gen_time_per_answer_ms=1000 * info["timing"][f"{suite}/{c}"]["per_answer_s"])
            for k in sorted(d.kind.unique()):
                r[f"acc_{k}"] = d[d.kind == k].score.mean()
            rows.append(r)
        for c in ("complementary", "mismatched", "no_image"):
            da = arr["correct"].score.to_numpy() - arr[c].score.to_numpy()
            dn = arr[c].nll.to_numpy() - arr["correct"].nll.to_numpy()
            dp = arr["correct"].both.to_numpy().astype(float) - arr[c].both.to_numpy().astype(float)
            r = dict(run=run, suite=suite, vs=c, acc_advantage_pp=100 * da.mean(),
                     acc_ci_lo_pp=100 * boot_ci(da, idx)[0], acc_ci_hi_pp=100 * boot_ci(da, idx)[1],
                     nll_advantage=dn.mean(), nll_ci_lo=boot_ci(dn, idx)[0], nll_ci_hi=boot_ci(dn, idx)[1],
                     pair_acc_advantage_pp=100 * dp.mean(), pair_acc_ci_lo_pp=100 * boot_ci(dp, idx)[0],
                     pair_acc_ci_hi_pp=100 * boot_ci(dp, idx)[1])
            for k in sorted(p.kind.unique()):
                m = (arr["correct"].kind == k).to_numpy()
                r[f"acc_advantage_pp_{k}"] = 100 * da[m].mean()
            adv.append(r)
        # how answers change under the complementary swap (original references fixed)
        cor = df[(df.suite == suite) & (df.condition == "correct")].set_index("qid")
        sw = df[(df.suite == suite) & (df.condition == "complementary")].set_index("qid").loc[cor.index]
        for k in ["all"] + sorted(cor.kind.unique()):
            m = np.ones(len(cor), bool) if k == "all" else (cor.kind == k).to_numpy()
            s, c0 = sw[m], cor[m]
            changed = (s.prediction_norm.to_numpy() != c0.prediction_norm.to_numpy())
            swap.append(dict(run=run, suite=suite, kind=k, n=int(m.sum()),
                             answer_changed=changed.mean(),
                             follows_partner=s.follows_partner.mean(),
                             keeps_original_reference=s.matches_canonical.mean(),
                             other_answer=(~s.follows_partner & ~s.matches_canonical & ~s.empty_answer).mean(),
                             empty=s.empty_answer.mean(),
                             changed_and_follows_partner=(changed & s.follows_partner.to_numpy()).mean(),
                             changed_to_unsupported=(changed & ~s.follows_partner.to_numpy()).mean(),
                             correct_on_original_then_follows_partner=(
                                 c0.matches_canonical.to_numpy() & s.follows_partner.to_numpy()).mean()))
    return pd.DataFrame(rows), pd.DataFrame(adv), pd.DataFrame(swap), pt


# ----------------------------------------------------------------------------- learning curves
def learning_curves(runs):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for run in runs:
        rd = C.run_dir(run)
        tl = pd.DataFrame(read_jsonl(rd / "train_log.jsonl"))
        dv = pd.DataFrame(read_jsonl(rd / "dev_log.jsonl")).sort_values("update")
        ls = "-" if run == "main" else "--"
        lab = f" ({run})" if len(runs) > 1 else ""
        ax = axes[0, 0]
        ax.plot(tl["update"], tl.loss.rolling(50, min_periods=1).mean(), ls, label="train loss (50-update mean)" + lab)
        ax = axes[0, 1]
        ax.plot(dv["update"], dv.caption_loss, "o" + ls, ms=3, label="dev caption loss" + lab)
        for j, suite in enumerate(("grounding", "vqa")):
            ax = axes[1, j]
            for cond, col in (("correct", "C0"), ("complementary", "C1"), ("no_image", "C2")):
                ax.plot(dv["update"], dv[f"{suite}_nll_{cond}"], "o" + ls, ms=3, color=col, label=f"{cond}{lab}")
            ax.set_title(f"dev {suite}: reference-answer NLL")
        ax = axes[0, 2]
        for suite, col in (("grounding", "C3"), ("vqa", "C4")):
            ax.plot(dv["update"], dv[f"{suite}_swap_penalty"], "o" + ls, ms=3, color=col, label=f"{suite}{lab}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("image-swap penalty  NLL(partner) - NLL(correct)")
        ax = axes[1, 2]
        for col in sorted(c for c in dv.columns if "/" in c and c.endswith("_swap_penalty")):  # e.g. vqa/yes/no
            ax.plot(dv["update"], dv[col], "o" + ls, ms=3, label=col[:-len("_swap_penalty")] + lab)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("swap penalty by question type (dev)")
    axes[0, 0].set_title("training loss")
    axes[0, 1].set_title("dev caption reference loss")
    for ax in axes.flat:
        for e in C.STAGE_ENDS[:-1]:
            ax.axvline(e, color="grey", ls=":", lw=1)
        ax.set_xlabel("update")
        ax.legend(fontsize=7)
    fig.suptitle("Stage boundaries (dotted): 2,250 caption alignment | 3,750 visual instruction tuning | 4,500 natural VQA")
    fig.tight_layout()
    fig.savefig(out_dir() / "learning_curves.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- qualitative
def _img(split, image_id):
    from PIL import Image
    p = C.image_path(split, image_id)
    return Image.open(p).convert("RGB") if p.exists() else None


def qualitative(df, run):
    """Selection rule: for each question type, the first final-test pair (in the seeded selection order)
    that the model gets fully right on the correct images, and the first where it gives the same answer
    to both images. Shows both images with the shared question, references and predictions."""
    cor = df[df.condition == "correct"]
    sw = df[df.condition == "complementary"].set_index("qid")
    picks = []
    for (suite, kind), g in cor.groupby(["suite", "kind"], sort=False):
        order = list(dict.fromkeys(g.pair_id))
        for label, test in (("both correct", lambda d: d.matches_canonical.all()),
                            ("same answer for both images", lambda d: d.prediction_norm.iloc[0] == d.prediction_norm.iloc[1])):
            for pid in order:
                d = g[g.pair_id == pid]
                if test(d):
                    picks.append((label, d))
                    break
    recs = []
    for label, d in picks:
        r = dict(rule=label, suite=d.suite.iloc[0], kind=d.kind.iloc[0], pair_id=d.pair_id.iloc[0],
                 question=d.question.iloc[0], images=[], references=[], pred_correct=[], pred_swap=[])
        for _, x in d.iterrows():
            r["images"].append(x.image)
            r["references"].append(x.target if x.suite == "grounding" else x.canonical)
            r["pred_correct"].append(x.prediction)
            r["pred_swap"].append(sw.loc[x.qid].prediction)
        recs.append(r)
    save_json(recs, out_dir() / f"qualitative_{run}.json")
    if not recs or _img(*recs[0]["images"][0]) is None:
        return recs
    n = len(recs)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.3 * n))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(recs):
        for j in range(2):
            ax = axes[i, j]
            ax.imshow(_img(*r["images"][j]))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel(textwrap.fill(f"ref: {r['references'][j]} | pred: {r['pred_correct'][j]!r} | "
                                        f"pred with other image: {r['pred_swap'][j]!r}", 55), fontsize=7)
        axes[i, 0].set_title(textwrap.fill(f"[{r['kind']} — {r['rule']}] {r['question']}", 60), fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir() / f"qualitative_{run}.png", dpi=110)
    plt.close(fig)
    return recs


def label_sheet(n=8):
    """Annotation-derived test labels shown with their images (no model involved), for manual inspection."""
    pairs = read_jsonl(C.data_dir() / "ground_test.jsonl")
    sel = [p for p in pairs if p["kind"] == "presence"][:n] + [p for p in pairs if p["kind"] == "counting"][:n]
    if _img(sel[0]["items"][0]["split"], sel[0]["items"][0]["image_id"]) is None:
        print("images not available; skipping label sheet")
        return
    fig, axes = plt.subplots(len(sel), 2, figsize=(6, 2.6 * len(sel)))
    for i, p in enumerate(sel):
        for j, it in enumerate(p["items"]):
            ax = axes[i, j]
            ax.imshow(_img(it["split"], it["image_id"]))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel(f"label: {it['target']}  (image {it['image_id']})", fontsize=7)
        axes[i, 0].set_title(p["question"], fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir() / "grounding_label_check.png", dpi=80)
    plt.close(fig)


# ----------------------------------------------------------------------------- resources
def resources(runs):
    rows = []
    fc = C.data_dir() / "feature_cache_info.json"
    fc = load_json(fc) if fc.exists() else {}
    for run in runs:
        res = load_json(C.run_dir(run) / "resources.json")
        for s, v in res["stages"].items():
            rows.append(dict(run=run, stage=s, updates=v["updates"], trainable_connector=v["trainable_connector"],
                             trainable_lora=v["trainable_lora"],
                             trainable_total=v["trainable_connector"] + v["trainable_lora"],
                             train_time_h=v["train_time_s"] / 3600,
                             s_per_update=v["train_time_s"] / v["updates"],
                             peak_gpu_mem_gb=v["peak_gpu_mem_allocated_bytes"] / 1e9))
        ev = C.run_dir(run) / "eval" / "stage3" / "eval_info.json"
        if ev.exists():
            e = load_json(ev)
            rows.append(dict(run=run, stage="final eval (all suites, 4 conditions)",
                             train_time_h=sum(t["total_s"] for t in e["timing"].values()) / 3600,
                             peak_gpu_mem_gb=e["peak_gpu_mem_allocated_bytes"] / 1e9))
    if fc:
        rows.append(dict(run="shared", stage=f"SigLIP feature caching ({fc.get('n_images')} images)",
                         train_time_h=fc.get("seconds", 0) / 3600,
                         peak_gpu_mem_gb=(fc.get("peak_gpu_mem_allocated_bytes") or 0) / 1e9))
    t = pd.DataFrame(rows)
    t.to_csv(out_dir() / "resources.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_only", action="store_true")
    a = ap.parse_args()
    label_sheet()
    if a.labels_only:
        return
    runs = sorted(p.name for p in (C.task_dir() / "runs").iterdir()
                  if (p / "train_log.jsonl").exists() and not p.name.endswith("_smoke")) or \
        sorted(p.name for p in (C.task_dir() / "runs").iterdir() if (p / "train_log.jsonl").exists())
    runs = sorted(runs, key=lambda r: r != "main")
    learning_curves(runs)
    summ, A, S, Q = [], [], [], {}
    for run in runs:
        ed = C.run_dir(run) / "eval" / "stage3"
        if not (ed / "predictions.jsonl").exists():
            continue
        df = pd.DataFrame(read_jsonl(ed / "predictions.jsonl"))
        t, adv, swap, pt = final_tables(run, df, load_json(ed / "eval_info.json"))
        summ.append(t), A.append(adv), S.append(swap)
        pt.to_csv(out_dir() / f"pairs_{run}.csv", index=False)
        Q[run] = qualitative(df, run)
    if summ:
        T, A, S = pd.concat(summ), pd.concat(A), pd.concat(S)
        T.to_csv(out_dir() / "final_results.csv", index=False)
        A.to_csv(out_dir() / "correct_image_advantage.csv", index=False)
        S.to_csv(out_dir() / "swap_behavior.csv", index=False)
        pd.set_option("display.width", 200)
        print(T[["run", "suite", "condition", "accuracy", "pair_accuracy", "identical_answer_rate", "ref_nll",
                 "empty_rate"]].round(3).to_string(index=False))
        print(A[["run", "suite", "vs", "acc_advantage_pp", "acc_ci_lo_pp", "acc_ci_hi_pp", "nll_advantage"]]
              .round(2).to_string(index=False))
    R = resources(runs)
    dev = {run: read_jsonl(C.run_dir(run) / "dev_log.jsonl") for run in runs}
    save_json(dict(runs=runs, final=T.to_dict("records") if summ else None,
                   advantage=A.to_dict("records") if summ else None, swap=S.to_dict("records") if summ else None,
                   resources=R.to_dict("records"), dev_last={r: d[-1] for r, d in dev.items() if d}),
              out_dir() / "summary.json")
    print(json.dumps(R.round(3).to_dict("records"), indent=1)[:2000])


if __name__ == "__main__":
    main()
