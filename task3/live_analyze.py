"""Analyze the live generation runs (PA3 Task 3, step 4).

    python -m task3.live_analyze      # reads outputs/task3/live/calls_shard*.jsonl -> summary.json, per_question.csv

Grading and the self-consistency vote over the three parallel answers use the same math-verify rules as the
offline MATH analysis (ties: earliest response). Costs charge the shared initial response to both the
parallel and the refinement strategy; the refinement cost includes the longer revision prompt and every
critique/revision token. Transitions compare the revised answer with the shared initial answer; gold answers
are used only for grading.
"""
import numpy as np
import pandas as pd

from common.utils import load_json, read_jsonl, save_json
from task3 import config as C
from task3.answers import math_groups
from task3.live import live_dir
from task3.selection import vote


def main():
    calls = []
    for p in sorted(live_dir().glob("calls_shard*.jsonl")):
        calls += read_jsonl(p)
    qs = {q["index"]: q for q in load_json(live_dir() / "questions.json")["questions"]}
    rows = []
    for i, q in qs.items():
        c = {(r["strategy"], r["call"]): r for r in calls if r["index"] == i}
        if len(c) < 5:
            print(f"question {i}: only {len(c)} of 5 calls, skipped")
            continue
        par = [c[("parallel", k)] for k in range(3)]
        single, rev = c[("single", 0)], c[("refine", 0)]
        answers = [single["extracted"]] + [p["extracted"] for p in par] + [rev["extracted"]]
        labels, gid, gcorr, indiv = math_groups(answers, q["gold"], {})
        g_par = gid[1:4]
        win, first, _, _ = vote(np.array(g_par))
        row = dict(index=i, sample_id=q["sample_id"], gold=q["gold"],
                   single_correct=bool(indiv[0]), parallel_correct=bool(gcorr[win] and win != 0),
                   initial_correct=bool(indiv[1]), refine_correct=bool(indiv[4]),
                   parallel_answer=labels[win], single_answer=answers[0], initial_answer=answers[1],
                   refine_answer=answers[4],
                   refine_changed_answer=bool(gid[4] != gid[1]))
        for name, cs in (("single", [single]), ("parallel", par), ("refine", [par[0], rev])):
            row[f"{name}_input_tokens"] = sum(x["n_input_tokens"] for x in cs)
            row[f"{name}_output_tokens"] = sum(x["n_output_tokens"] for x in cs)
            row[f"{name}_time_s"] = sum(x["time_s"] for x in cs)
            row[f"{name}_peak_mem_gb"] = max(x["peak_gpu_mem_bytes"] for x in cs) / 1e9
            row[f"{name}_any_cap"] = any(x["hit_cap"] for x in cs)
        row["single_cap"], row["single_no_extract"] = single["hit_cap"], single["extracted"] is None
        row["parallel_cap_rate"] = np.mean([p["hit_cap"] for p in par])
        row["parallel_no_extract_rate"] = np.mean([p["extracted"] is None for p in par])
        row["initial_cap"], row["initial_no_extract"] = par[0]["hit_cap"], par[0]["extracted"] is None
        row["revision_cap"], row["revision_no_extract"] = rev["hit_cap"], rev["extracted"] is None
        rows.append(row)
    d = pd.DataFrame(rows).sort_values("index")
    d.to_csv(live_dir() / "per_question.csv", index=False)

    rng = np.random.default_rng(C.SEED)

    def boot(a, b):
        x = (d[a].astype(float) - d[b].astype(float)).to_numpy()
        idx = rng.integers(0, len(x), size=(C.N_BOOT, len(x)))
        bs = x[idx].mean(1)
        return dict(diff_pp=100 * x.mean(), ci95_pp=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)])

    strat = {}
    for name in ("single", "parallel", "refine"):
        strat[name] = dict(accuracy=float(d[f"{name}_correct"].mean()),
                           mean_input_tokens=float(d[f"{name}_input_tokens"].mean()),
                           mean_output_tokens=float(d[f"{name}_output_tokens"].mean()),
                           mean_runtime_s=float(d[f"{name}_time_s"].mean()),
                           total_runtime_s=float(d[f"{name}_time_s"].sum()),
                           peak_gpu_mem_gb=float(d[f"{name}_peak_mem_gb"].max()),
                           questions_with_any_capped_call=float(d[f"{name}_any_cap"].mean()))
    strat["single"].update(cap_rate=float(d.single_cap.mean()), no_extract_rate=float(d.single_no_extract.mean()))
    strat["parallel"].update(cap_rate_per_response=float(d.parallel_cap_rate.mean()),
                             no_extract_rate_per_response=float(d.parallel_no_extract_rate.mean()))
    strat["refine"].update(initial_cap_rate=float(d.initial_cap.mean()), revision_cap_rate=float(d.revision_cap.mean()),
                           revision_no_extract_rate=float(d.revision_no_extract.mean()),
                           initial_accuracy=float(d.initial_correct.mean()))
    trans = dict(wrong_to_correct=int((~d.initial_correct & d.refine_correct).sum()),
                 correct_to_wrong=int((d.initial_correct & ~d.refine_correct).sum()),
                 correct_to_correct=int((d.initial_correct & d.refine_correct).sum()),
                 wrong_to_wrong=int((~d.initial_correct & ~d.refine_correct).sum()),
                 answer_changed=int(d.refine_changed_answer.sum()))
    summary = dict(n_questions=len(d), strategies=strat, refinement_transitions=trans,
                   differences=dict(parallel_vs_single=boot("parallel_correct", "single_correct"),
                                    refine_vs_single=boot("refine_correct", "single_correct"),
                                    refine_vs_parallel=boot("refine_correct", "parallel_correct"),
                                    refine_vs_initial=boot("refine_correct", "initial_correct")),
                   run_info=[load_json(p) for p in sorted(live_dir().glob("run_info_shard*.json"))])
    save_json(summary, live_dir() / "summary.json")
    print(pd.DataFrame(strat).T.round(3).to_string())
    print(trans)


if __name__ == "__main__":
    main()
