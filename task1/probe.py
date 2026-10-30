"""Activation-probe set: 32 GSM8K test questions + 32 HellaSwag validation contexts, sampled with seed
6304 *before* looking at any model results. Prompts use the same formats as the benchmarks:
GSM8K -> gsm8k_cot doc_to_text inside the non-thinking chat template; HellaSwag -> stock completion
context (no chat template).

    python -m task1.probe          -> outputs/task1/probe_set.json
"""
import numpy as np

from common.utils import load_json, save_json
from task1 import config as C
from task1.data import prompt_text
from task1.modeling import load_tokenizer


def _task(name):
    from lm_eval.tasks import TaskManager, get_task_dict

    return get_task_dict([name], TaskManager())[name]


def build():
    rng = np.random.default_rng(C.SEED)
    tok = load_tokenizer()
    gsm_task, hs_task = _task(C.GSM8K_TASK), _task("hellaswag")
    gsm_docs, hs_docs = list(gsm_task.test_docs()), list(hs_task.validation_docs())
    gi = sorted(rng.choice(len(gsm_docs), C.N_PROBE_PER_SET, replace=False).tolist())
    hi = sorted(rng.choice(len(hs_docs), C.N_PROBE_PER_SET, replace=False).tolist())
    items = []
    for i in gi:
        items.append(dict(set="gsm8k", doc_id=i, kind="math",
                          prompt=prompt_text(tok, gsm_task.doc_to_text(gsm_docs[i]))))
    for i in hi:
        items.append(dict(set="hellaswag", doc_id=i, kind="non_math", prompt=hs_task.doc_to_text(hs_docs[i])))
    out = C.task_dir() / "probe_set.json"
    save_json(dict(seed=C.SEED, gsm8k_test_ids=gi, hellaswag_val_ids=hi, items=items), out)
    print(f"wrote {out}")
    return out


def load():
    p = C.task_dir() / "probe_set.json"
    if not p.exists():
        build()
    return load_json(p)


if __name__ == "__main__":
    build()
