"""Task 1 configuration. Every reported hyperparameter lives here so runs are reproducible."""
from common.utils import output_root

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 6304

# ---- data ----
METAMATH_ID = "meta-math/MetaMathQA"
GSM8K_ID = "openai/gsm8k"
N_TRAIN = 40_000
MAX_LEN = 2048  # formatted (prompt + response + <|im_end|>) length cap

# ---- shared training recipe ----
MICRO_BS = 2
GRAD_ACCUM = 16
EFF_BS = MICRO_BS * GRAD_ACCUM  # 32
TOTAL_UPDATES = N_TRAIN // EFF_BS  # 1250
WARMUP_UPDATES = 38
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
CE_CHUNK = 1024  # tokens per checkpointed lm_head/CE chunk (memory only, no effect on maths)

# Recovery checkpoints (full resumable state) and evaluation exports, in optimizer updates.
RECOVERY_UPDATES = [312, 625, 937, 1250]
EVAL_UPDATES = {
    "full_ft": [625, 1250],
    "full_ft_l2sp": [1250],
    "lora4": [1250],
    "lora16": [1250],
    "oplora16": [1250],
}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

METHODS = {
    "full_ft": dict(kind="full", lr=2e-5, l2sp_lambda=0.0),
    "full_ft_l2sp": dict(kind="full", lr=2e-5, l2sp_lambda=1e-3),
    "lora4": dict(kind="lora", lr=2e-4, r=4, alpha=8, dropout=0.05),
    "lora16": dict(kind="lora", lr=2e-4, r=16, alpha=32, dropout=0.05),
    "oplora16": dict(kind="oplora", lr=2e-4, r=16, alpha=32, dropout=0.05, k=128),
}

# ---- evaluation ----
GSM8K_TASK = "gsm8k_cot"
GSM8K_MAX_NEW = 1024
LL_TASKS = ["hellaswag", "boolq", "arc_easy"]
PRIMARY_METRIC = {
    "gsm8k_cot": "exact_match,flexible-extract",
    "hellaswag": "acc_norm,none",
    "boolq": "acc,none",
    "arc_easy": "acc_norm,none",
}

# Seven evaluation conditions: (name, method, update)
CONDITIONS = [
    ("baseline", None, 0),
    ("full_ft@625", "full_ft", 625),
    ("full_ft", "full_ft", 1250),
    ("full_ft_l2sp", "full_ft_l2sp", 1250),
    ("lora4", "lora4", 1250),
    ("lora16", "lora16", 1250),
    ("oplora16", "oplora16", 1250),
]

# ---- diagnostics ----
DIAG_LAYERS = [0, 13, 27]
DIAG_K = 128
N_PROBE_PER_SET = 32
QUAL_MAX_NEW = 512


def task_dir():
    return output_root() / "task1"


def run_dir(method):
    return task_dir() / "runs" / method
