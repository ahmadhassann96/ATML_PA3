"""Download everything Task 1 needs once (model, MetaMathQA, the four lm-eval benchmarks, the
qualitative prompt file), so later runs
can use HF_HUB_OFFLINE=1 and never block on the network.

    python -m task1.prefetch
"""
import urllib.request
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from task1 import config as C


QUAL_URL = ("https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/"
            "evaluation/first_plot_questions.yaml")  # linked from the PA (Task 1, qualitative inputs)


def main():
    if not Path("first_plot_questions.yaml").exists():
        urllib.request.urlretrieve(QUAL_URL, "first_plot_questions.yaml")
        print("downloaded first_plot_questions.yaml")
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
