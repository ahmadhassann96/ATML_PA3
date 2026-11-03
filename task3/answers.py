"""Answer normalization, answer groups and correctness labels (scorer pa3-task3-scorer-v1).

    python -m task3.answers          # -> outputs/task3/answers/<ds>.parquet + <ds>_summary.json

Groups are formed per question from the extracted answers alone (gold answers are never consulted),
in order of first appearance by rollout ID. A missing extraction, or one that is unusable under the
dataset's answer rules, joins a single NO_ANSWER group (group 0) that is always graded incorrect.
Correctness is then computed for each group with the same equivalence rule.

MATH: math-verify equivalence. Each answer is parsed as \\boxed{<answer>}; a new answer joins the first
existing group whose representative verifies against it, otherwise it opens a new group. A non-empty
answer that math-verify cannot parse stays a group of its own (keyed by its exact text).

MathVista: the released MathVista rules (normalize_extracted_answer in MathVista's calculate_score.py)
using testmini.json metadata: multiple choice -> option letter (direct letter, "(X)", otherwise the most
similar option text by edit distance); integer -> str(int(float(x))); float -> str(round(float(x),
precision)); list -> "[a, b, ...]". Before int/float parsing a light LaTeX clean-up removes formatting the
export's boxed extractor keeps but MathVista's plain-text rules cannot read (\\text{unit}, \\mathrm{},
thin spaces, $, %, degree signs, thousands separators, \\frac{a}{b}); anything still not a number is unusable.
Gold answers pass through the same normalization.
"""
import argparse
import re

import numpy as np
import pandas as pd

from common.utils import load_json, save_json
from task3 import config as C

NO_ANSWER = "NO_ANSWER"
MAX_PARSE_CHARS = 400   # longer strings are kept as exact-text groups (math-verify runs without a timeout here)


# ----------------------------------------------------------------------------------------------- MATH
def _mv():
    from math_verify import parse, verify
    import logging
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    return parse, verify


def math_groups(answers, gold, cache, skip=()):
    """answers: list of 32 extracted strings/None (rollout-ID order). Returns (labels, group_ids,
    group_correct, individual_correct). cache: dict for parsed expressions. skip: strings that are not
    parsed (kept as exact-text answers) because parsing them did not finish within the time limit."""
    parse, verify = _mv()

    def P(s):
        if s not in cache:
            ok = len(s) <= MAX_PARSE_CHARS and s not in skip
            cache[s] = parse("\\boxed{" + s + "}", parsing_timeout=None) if ok else []
        return cache[s]

    def eq(a, b):
        pa, pb = P(a), P(b)
        if not pa or not pb:
            return a == b
        return bool(verify(pa, pb, timeout_seconds=None))

    reps = []           # representative string per group (group id = index + 1)
    gid = []
    for a in answers:
        if a is None or not str(a).strip():
            gid.append(0)
            continue
        a = str(a).strip()
        for j, rep in enumerate(reps):
            if a == rep or eq(rep, a):
                gid.append(j + 1)
                break
        else:
            reps.append(a)
            gid.append(len(reps))
    g = str(gold).strip()
    group_correct = [False] + [eq(g, rep) for rep in reps]
    indiv = [False if (a is None or not str(a).strip()) else eq(g, str(a).strip()) for a in answers]
    labels = [NO_ANSWER] + reps
    return labels, np.array(gid), np.array(group_correct), np.array(indiv)


QUESTION_TIMEOUT = 15   # seconds for grouping + grading one question (one worker, so it starts on time)
PARSE_TIMEOUT = 15      # seconds for parsing one answer when a question timed out


def _math_task(task, skip=()):
    q, answers, gold = task
    labels, gid, gc, indiv = math_groups(answers, gold, {}, skip=set(skip))
    return q, labels, gid.tolist(), gc.tolist(), indiv.tolist()


def _parse_task(s):
    parse, _ = _mv()
    parse("\\boxed{" + s + "}", parsing_timeout=None)
    return s


def _run_with_timeouts(tasks, fn, timeout, workers=1, extra=()):
    """Run fn(task, *extra) in worker processes; a task that exceeds `timeout` is reported as None.
    The pool is restarted after a timeout, so a runaway expression cannot keep growing."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results, pending = {}, list(tasks)
    while pending:
        pool = ctx.Pool(workers)
        pool.apply(_mv)   # import math-verify in the worker before any task is timed
        handles = [(t, pool.apply_async(fn, (t, *extra))) for t in pending]
        failed = None
        for n, (t, h) in enumerate(handles):
            if n % 100 == 0 and len(handles) > 100:
                print(f"  {len(results)}/{len(tasks)} done", flush=True)
            try:
                results[_key(t)] = h.get(timeout=timeout)
            except mp.TimeoutError:
                failed = t
                break
        pool.terminate()
        pool.join()
        if failed is not None:
            results[_key(failed)] = None
            print(f"  timeout: {str(_key(failed))[:80]!r}", flush=True)
        pending = [t for t in pending if _key(t) not in results]
    return results


def _key(t):
    return t[0] if isinstance(t, tuple) else t


def math_all(meta):
    """Group and grade every MATH question; questions that exceed the time limit are redone with the
    answers that cannot be parsed in time treated as exact-text answers."""
    tasks = []
    for q, g in meta.groupby("q", sort=True):
        g = g.sort_values("rollout_id")
        tasks.append((int(q), [None if pd.isna(a) else str(a) for a in g.extracted_answer], g.gold_answer.iloc[0]))
    res = _run_with_timeouts(tasks, _math_task, QUESTION_TIMEOUT)
    slow = {}
    for t in tasks:
        if res[t[0]] is None:
            uniq = sorted({a.strip() for a in t[1] if a and a.strip()} | {str(t[2]).strip()})
            pr = _run_with_timeouts(uniq, _parse_task, PARSE_TIMEOUT)
            skip = [s for s, v in pr.items() if v is None]
            slow[t[0]] = skip
            again = _run_with_timeouts([t], _math_task, QUESTION_TIMEOUT * 3, extra=(skip,))
            res[t[0]] = again[t[0]]
            if res[t[0]] is None:   # last resort: exact-text grouping and grading for this question
                ans = [a.strip() if a and a.strip() else None for a in t[1]]
                labels = [NO_ANSWER] + list(dict.fromkeys(a for a in ans if a))
                gid = [0 if a is None else labels.index(a) for a in ans]
                gc = [False] + [lab == str(t[2]).strip() for lab in labels[1:]]
                res[t[0]] = (t[0], labels, gid, gc, [gc[k] for k in gid])
                slow[t[0]].append("EXACT_TEXT_FALLBACK")
    return res, slow


# ------------------------------------------------------------------------------------------ MathVista
def _levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _strip_latex(s):
    s = str(s).strip()
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\[,;:! ]|~", " ", s)
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\^\s*\{?\\circ\}?|°|\\degree", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _to_number(s):
    """Numeric value of a cleaned answer, or None. Units written as words are dropped only when the rest
    is a single number (e.g. '0.68 minutes')."""
    s = _strip_latex(s)
    s = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    m = re.fullmatch(r"(-?\(?-?\d+(?:\.\d+)?\)?(?:\s*/\s*\(?-?\d+(?:\.\d+)?\)?)?)\s*([A-Za-z][A-Za-z/ ^0-9.]*)?", s)
    if not m:
        return None
    expr = m.group(1).replace(" ", "")
    if not re.fullmatch(r"[-0-9.()/]+", expr):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - digits, dots, parentheses and '/' only
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def mathvista_normalize(extraction, meta):
    """MathVista answer rule -> normalized string, or None if missing/unusable."""
    if extraction is None or not str(extraction).strip():
        return None
    qt, at, prec, choices = meta["question_type"], meta["answer_type"], meta["precision"], meta["choices"]
    if qt == "multi_choice":
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        e = _strip_latex(extraction)
        e1 = re.sub(r"^[\(\[]?([A-Za-z])[\)\]]?[.:]?$", r"\1", e)
        if e1.upper() in letters and len(e1) == 1:
            return e1.upper()
        m = re.findall(r"\(([a-zA-Z])\)", e)
        if m and m[0].upper() in letters:
            return m[0].upper()
        d = [_levenshtein(e, str(c)) for c in choices]   # MathVista: most similar option
        return letters[int(np.argmin(d))]
    if at == "integer":
        v = _to_number(extraction)
        return None if v is None else str(int(v))
    if at == "float":
        v = _to_number(extraction)
        return None if v is None else str(round(v, int(prec)))
    if at == "list":
        nums = re.findall(r"-?\d+(?:\.\d+)?", _strip_latex(extraction))
        return "[" + ", ".join(nums) + "]" if nums else None
    return str(extraction).strip()


def mathvista_gold(gold, meta):
    if meta["question_type"] == "multi_choice":
        return str(gold).strip().upper()
    return mathvista_normalize(gold, meta)


# ------------------------------------------------------------------------------------------------ run
def build(ds):
    meta = pd.read_parquet(C.cache_dir() / f"{ds}_rollouts.parquet")
    rows, stats = [], dict(scorer=C.SCORER_VERSION)
    if ds == "math":
        import importlib.metadata as im
        stats["math_verify_version"] = im.version("math-verify")
        res, slow = math_all(meta)
        mismatch = 0
        for q, g in meta.groupby("q", sort=True):
            g = g.sort_values("rollout_id")
            ans = [None if pd.isna(a) else a for a in g.extracted_answer]
            _, labels, gid, gc, indiv = res[int(q)]
            gc = np.array(gc)
            mismatch += int((gc[gid] != np.array(indiv)).sum())
            for r, a, k, ic in zip(g.rollout_id, ans, gid, indiv):
                rows.append(dict(q=q, sample_id=g.sample_id.iloc[0], rollout_id=r, extracted=a,
                                 norm=labels[k], group=int(k), correct=bool(gc[k]), individual_correct=bool(ic)))
        stats["group_vs_individual_grading_mismatches"] = mismatch
        stats["time_limited_questions"] = {meta[meta.q == q].sample_id.iloc[0]: s for q, s in slow.items()}
        stats["time_limits_s"] = dict(question=QUESTION_TIMEOUT, parse=PARSE_TIMEOUT)
    else:
        mv = load_json(C.MATHVISTA_META)
        stats["metadata"] = str(C.MATHVISTA_META)
        for q, g in meta.groupby("q", sort=True):
            g = g.sort_values("rollout_id")
            m = mv[g.sample_id.iloc[0]]
            gold = mathvista_gold(g.gold_answer.iloc[0], m)
            norms = [mathvista_normalize(None if pd.isna(a) else a, m) for a in g.extracted_answer]
            labels, gid = [NO_ANSWER], []
            for n in norms:
                if n is None:
                    gid.append(0)
                    continue
                if n not in labels:
                    labels.append(n)
                gid.append(labels.index(n))
            for r, a, n, k in zip(g.rollout_id, g.extracted_answer, norms, gid):
                rows.append(dict(q=q, sample_id=g.sample_id.iloc[0], rollout_id=r,
                                 extracted=None if pd.isna(a) else a, norm=labels[k], group=int(k),
                                 correct=bool(k != 0 and labels[k] == gold), gold_norm=gold,
                                 question_type=m["question_type"], answer_type=m["answer_type"]))
    df = pd.DataFrame(rows)
    out = C.task_dir() / "answers"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{ds}.parquet")
    stats.update(
        rollouts=len(df), missing_extraction=int(df.extracted.isna().sum()),
        no_answer_group=int((df.group == 0).sum()),
        unusable_nonempty=int(((df.group == 0) & df.extracted.notna()).sum()),
        rollout_accuracy=float(df.correct.mean()),
        mean_groups_per_question=float(df.groupby("q").group.nunique().mean()),
        questions_with_any_correct=float(df.groupby("q").correct.any().mean()))
    if ds == "mathvista":
        stats["by_answer_type"] = {f"{k[0]}/{k[1]}": dict(n_rollouts=len(v), rollout_acc=float(v.correct.mean()),
                                                         no_answer=float((v.group == 0).mean()))
                                   for k, v in df.groupby(["question_type", "answer_type"])}
    save_json(stats, out / f"{ds}_summary.json")
    print(ds, stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    for d in ap.parse_args().datasets:
        build(d)
