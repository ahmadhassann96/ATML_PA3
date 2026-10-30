"""Qualitative generations: 24 prompts (8 questions x plain/JSON/template) x 7 conditions = 168 responses.
Greedy, thinking disabled, 512-new-token cap.

    python -m task1.qualitative --yaml first_plot_questions.yaml --condition baseline   (repeat per condition)
    python -m task1.qualitative --yaml first_plot_questions.yaml --collect

--collect groups all responses by question and format and writes the 42-row annotation sheet
(first and last question in plain-question order x 3 formats x 7 conditions) with automatic helper
columns (truncation, JSON validity, repetition, math-framing cues). Judgement columns are left blank.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
import yaml

from common.utils import read_jsonl, save_json, write_jsonl
from task1 import config as C
from task1.data import prompt_text
from task1.modeling import load_condition, load_tokenizer

FORMATS = ["plain", "json", "template"]


def qual_dir():
    import task1.modeling as M
    return C.task_dir() / ("qualitative_smoke" if M.SMOKE else "qualitative")


def load_prompts(path):
    """Return [{qid, q_index, format, prompt}] in the file's plain-question order.
    Accepts the common layouts: a list of question records with one field per format, a mapping
    format -> list of prompts, or a flat list of {format, prompt} records."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and not any(k in raw for k in FORMATS):
        lists = [v for v in raw.values() if isinstance(v, list)]
        if len(lists) == 1:
            raw = lists[0]
    out = []

    def fmt_key(k):
        k = str(k).lower()
        for f in FORMATS:
            if f in k:
                return f
        return None

    if isinstance(raw, dict):  # format -> list
        per = {fmt_key(k): v for k, v in raw.items() if fmt_key(k)}
        n = len(per["plain"])
        for i in range(n):
            for f in FORMATS:
                out.append(dict(qid=str(i), q_index=i, format=f, prompt=_text(per[f][i])))
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict) and any(fmt_key(k) for k in raw[0]):
        if "format" in raw[0]:  # flat list of {format, prompt}
            order = {}
            for r in raw:
                qid = str(r.get("id", r.get("question_id", r.get("qid"))))
                order.setdefault(qid, len(order))
                out.append(dict(qid=qid, q_index=order[qid], format=fmt_key(r["format"]),
                                prompt=_text(r.get("prompt", r.get("question", r.get("text"))))))
        else:  # one record per question
            for i, r in enumerate(raw):
                qid = str(r.get("id", r.get("question_id", i)))
                for k, v in r.items():
                    f = fmt_key(k)
                    if f:
                        out.append(dict(qid=qid, q_index=i, format=f, prompt=_text(v)))
    else:
        raise ValueError(f"unrecognised YAML layout in {path}; top-level type {type(raw)}")
    assert len(out) == 24, f"expected 24 prompts, parsed {len(out)} -- check load_prompts against the file"
    return sorted(out, key=lambda r: (r["q_index"], FORMATS.index(r["format"])))


def _text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("prompt", "question", "text", "content"):
            if k in v:
                return v[k]
    raise ValueError(f"cannot read prompt from {v!r}")


@torch.no_grad()
def generate(condition, prompts, bs=8):
    tok = load_tokenizer()
    tok.padding_side = "left"
    model = load_condition(condition)
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for s in range(0, len(prompts), bs):
        chunk = prompts[s : s + bs]
        texts = [prompt_text(tok, p["prompt"]) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=C.QUAL_MAX_NEW, do_sample=False, temperature=None,
                             top_p=None, top_k=None, eos_token_id=[eot, tok.eos_token_id],
                             pad_token_id=tok.pad_token_id)
        for p, g in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            g = g.tolist()
            ended = any(t in (eot, tok.eos_token_id) for t in g)
            n = next((i for i, t in enumerate(g) if t in (eot, tok.eos_token_id)), len(g))
            rows.append(dict(condition=condition, **p, response=tok.decode(g[:n], skip_special_tokens=True),
                             n_new_tokens=n, truncated=not ended))
    out = qual_dir() / f"{condition.replace('@', '_at_')}.jsonl"
    write_jsonl(rows, out)
    print(f"wrote {out}")


def _repetition(text, n=4):
    toks = text.split()
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def _json_ok(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        json.loads(t)
        return True
    except Exception:  # noqa: BLE001
        return False


def collect():
    qd = qual_dir()
    rows = []
    for cname, _, _ in C.CONDITIONS:
        f = qd / f"{cname.replace('@', '_at_')}.jsonl"
        if f.exists():
            rows += read_jsonl(f)
        else:
            print(f"missing {f}")
    grouped = {}
    for r in rows:
        grouped.setdefault(f"q{r['q_index']}", {}).setdefault(r["format"], {})[r["condition"]] = dict(
            response=r["response"], truncated=r["truncated"], n_new_tokens=r["n_new_tokens"], prompt=r["prompt"])
    save_json(grouped, qd / "all_responses_grouped.json")

    qs = sorted({r["q_index"] for r in rows})
    sel = [r for r in rows if r["q_index"] in (qs[0], qs[-1])]
    order = [c[0] for c in C.CONDITIONS]
    sel.sort(key=lambda r: (r["q_index"], FORMATS.index(r["format"]), order.index(r["condition"])))
    df = pd.DataFrame([dict(
        q_index=r["q_index"], format=r["format"], condition=r["condition"], truncated=r["truncated"],
        n_new_tokens=r["n_new_tokens"], json_parses=_json_ok(r["response"]) if r["format"] == "json" else None,
        repeated_4gram_frac=round(_repetition(r["response"]), 3),
        math_cues=len(re.findall(r"\\boxed|\\frac|\$|=|step \d", r["response"], flags=re.I)),
        format_compliance="", coherence_relevance="", repetition="", unnecessary_math_framing="",
        inappropriate_content="", excerpt="", notes="", response=r["response"]) for r in sel])
    df.to_csv(qd / "annotation_sheet_42.csv", index=False, encoding="utf-8")
    print(f"{len(rows)} responses collected; annotation sheet has {len(df)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="first_plot_questions.yaml")
    ap.add_argument("--condition", choices=[c[0] for c in C.CONDITIONS])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="use the tiny *_smoke training runs")
    a = ap.parse_args()
    if a.smoke:
        import task1.modeling as M
        M.SMOKE = True
    if a.collect:
        collect()
    else:
        generate(a.condition, load_prompts(a.yaml), a.bs)
