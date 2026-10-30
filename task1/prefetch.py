"""Download everything Task 1 needs once (model, MetaMathQA, the four lm-eval benchmarks), so later runs
can use HF_HUB_OFFLINE=1 and never block on the network.

    python -m task1.prefetch
"""
from datasets import load_dataset
from huggingface_hub import snapshot_download

from task1 import config as C


def main():
    print("model:", snapshot_download(C.MODEL_ID))
    load_dataset(C.METAMATH_ID, split="train")
    load_dataset(C.GSM8K_ID, "main", split="test")
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = get_task_dict([C.GSM8K_TASK] + C.LL_TASKS, TaskManager())
    for name, t in tasks.items():
        docs = t.test_docs() if t.has_test_docs() else t.validation_docs()
        print(f"{name}: {len(list(docs))} eval docs")
    print("done -- you can now set HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
