"""Task 3 settings (PA3 section 3)."""
import os
from pathlib import Path

from common.utils import output_root

SEED = 6304
DATA_ROOT = Path(os.environ.get("PA3_DATA", "ATML-PA3-Data"))
MATHVISTA_META = Path(os.environ.get("PA3_MATHVISTA_META", "data/mathvista_testmini.json"))

DATASETS = {
    "math": dict(folder="qwen25-math", vlm=False, quality="selfcert", label="MATH (Qwen2.5-3B)"),
    "mathvista": dict(folder="qwen25vl-mathvista", vlm=True, quality="visconf", label="MathVista (Qwen2.5-VL-3B)"),
}
N_ROLLOUTS = 32
BUDGETS = [1, 2, 4, 8, 16, 32]
N_ORDERS = 10
N_DEV = 200
N_BOOT = 1000

# VisConf calibration percentiles and denominator floor
CAL_LO, CAL_HI, CAL_EPS = 5, 95, 1e-8

# Adaptive stopping (step 5)
CHECKPOINTS = [4, 8, 16]
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
DEV_TOLERANCE = 0.01  # accuracy may drop at most 1 pp below fixed-32 self-consistency on dev

# Live generation (step 4)
LIVE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LIVE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LIVE_N_QUESTIONS = 50
SYSTEM_MSG = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW_MSG = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
              "Give your final answer inside \\boxed{}.")
GEN_KW = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, do_sample=True)
LIVE_LIMITS = dict(single=3072, parallel=1024, n_parallel=3, revision=2048)

SCORER_VERSION = "pa3-task3-scorer-v1"


def task_dir() -> Path:
    return output_root() / "task3"


def cache_dir() -> Path:
    d = task_dir() / "cache"   # git-ignored
    d.mkdir(parents=True, exist_ok=True)
    return d
