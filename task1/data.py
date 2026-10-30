"""Build the MetaMathQA SFT set exactly as the PA specifies.

- first 40,000 *eligible* examples in source order (no shuffling, no answer truncation)
- drop: empty query/response, formatted length > 2048 tokens,
        query or original_question matching a GSM8K test question after normalization
- format: user question -> assistant response, thinking disabled
- labels: only assistant-response tokens + the <|im_end|> end-of-turn token are supervised

Usage:  python -m task1.data [--n 40000]
Writes  outputs/task1/data/train.npz (token ids + offsets) and selection.json (source row ids, stats).
"""
import argparse
import re
import unicodedata

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from common.utils import load_json, save_json
from task1 import config as C

END_OF_TURN = "<|im_end|>"


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def prompt_text(tokenizer, question: str) -> str:
    """Chat prompt for one user question, thinking disabled; ends with the assistant header
    (Qwen3 inserts an empty <think></think> block when enable_thinking=False)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def check_template(tokenizer) -> None:
    """Our prompt+response split must reproduce the official full-conversation rendering."""
    q, a = "What is 2+3?", "2+3=5. The answer is 5."
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        tokenize=False,
        enable_thinking=False,
    )
    ours = prompt_text(tokenizer, q) + a + END_OF_TURN
    assert full.startswith(ours), f"template mismatch:\n{full!r}\nvs\n{ours!r}"


def gsm8k_test_questions() -> set:
    ds = load_dataset(C.GSM8K_ID, "main", split="test")
    return {normalize_text(q) for q in ds["question"]}


def build(n_target: int = C.N_TRAIN, batch: int = 512):
    tok = AutoTokenizer.from_pretrained(C.MODEL_ID)
    check_template(tok)
    eot_id = tok.convert_tokens_to_ids(END_OF_TURN)
    gsm = gsm8k_test_questions()

    src = load_dataset(C.METAMATH_ID, split="train")  # source order
    stats = dict(scanned=0, empty=0, gsm8k_overlap=0, too_long=0, kept=0)
    kept_rows, all_ids, prompt_lens, total_lens = [], [], [], []

    pbar = tqdm(total=n_target, desc="selecting")
    for start in range(0, len(src), batch):
        if len(kept_rows) >= n_target:
            break
        rows = src[start : start + batch]
        cand = []
        for j in range(len(rows["query"])):
            q, r = rows["query"][j], rows["response"][j]
            stats["scanned"] += 1
            if not (q or "").strip() or not (r or "").strip():
                stats["empty"] += 1
                continue
            if normalize_text(q) in gsm or normalize_text(rows["original_question"][j]) in gsm:
                stats["gsm8k_overlap"] += 1
                continue
            cand.append((start + j, q, r))
        if not cand:
            continue
        p_ids = tok([prompt_text(tok, q) for _, q, _ in cand], add_special_tokens=False)["input_ids"]
        r_ids = tok([r for _, _, r in cand], add_special_tokens=False)["input_ids"]
        for (row, _, _), p, r in zip(cand, p_ids, r_ids):
            if len(kept_rows) >= n_target:
                break
            ids = p + r + [eot_id]
            if len(ids) > C.MAX_LEN:
                stats["too_long"] += 1
                continue
            kept_rows.append(row)
            all_ids.append(np.asarray(ids, dtype=np.int32))
            prompt_lens.append(len(p))
            total_lens.append(len(ids))
            pbar.update(1)
    pbar.close()
    stats["kept"] = len(kept_rows)
    stats["scanned"] = kept_rows[-1] + 1  # rows up to and including the last kept one
    assert len(kept_rows) == n_target, f"only {len(kept_rows)} eligible examples"

    offsets = np.zeros(len(all_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(x) for x in all_ids])
    out = C.task_dir() / "data"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "train.npz",
        tokens=np.concatenate(all_ids),
        offsets=offsets,
        prompt_lens=np.asarray(prompt_lens, dtype=np.int32),
    )
    tl = np.asarray(total_lens)
    resp = tl - np.asarray(prompt_lens)
    save_json(
        dict(
            source=C.METAMATH_ID,
            model=C.MODEL_ID,
            n=len(kept_rows),
            last_source_row=kept_rows[-1],
            stats=stats,
            normalization="NFKC, lowercase, non-alphanumerics -> space, collapse whitespace",
            token_stats=dict(
                total_tokens=int(tl.sum()),
                supervised_tokens=int(resp.sum()),
                mean_len=float(tl.mean()),
                max_len=int(tl.max()),
                mean_resp_len=float(resp.mean()),
            ),
            source_rows=kept_rows,
        ),
        out / "selection.json",
    )
    print(stats)
    return out


class SFTData:
    """Fixed-order view of the tokenized set; example i is always the i-th selected row."""

    def __init__(self, path=None, limit=None):
        path = path or (C.task_dir() / "data" / "train.npz")
        z = np.load(path)
        self.tokens, self.offsets, self.prompt_lens = z["tokens"], z["offsets"], z["prompt_lens"]
        self.n = len(self.prompt_lens) if limit is None else min(limit, len(self.prompt_lens))

    def __len__(self):
        return self.n

    def example(self, i):
        ids = self.tokens[self.offsets[i] : self.offsets[i + 1]].astype(np.int64)
        labels = ids.copy()
        labels[: self.prompt_lens[i]] = -100  # mask prompt (incl. chat header + empty think block)
        return ids, labels

    def microbatch(self, indices, pad_id):
        import torch

        exs = [self.example(i) for i in indices]
        T = max(len(x) for x, _ in exs)
        ids = torch.full((len(exs), T), pad_id, dtype=torch.long)
        lab = torch.full((len(exs), T), -100, dtype=torch.long)
        att = torch.zeros((len(exs), T), dtype=torch.long)
        for b, (x, y) in enumerate(exs):
            ids[b, : len(x)] = torch.from_numpy(x)
            lab[b, : len(y)] = torch.from_numpy(y)
            att[b, : len(x)] = 1
        return ids, att, lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N_TRAIN)
    args = ap.parse_args()
    build(args.n)
    print(load_json(C.task_dir() / "data" / "selection.json")["token_stats"])
