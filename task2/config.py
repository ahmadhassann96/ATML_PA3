"""Task 2 configuration (SigLIP -> MLP connector -> Qwen3-0.6B). Every reported setting lives here."""
import os
from pathlib import Path

from common.utils import output_root

SEED = 6304
LM_ID = "Qwen/Qwen3-0.6B"
VISION_ID = "google/siglip-base-patch16-224"
N_VIS, D_VIS, D_LM = 196, 768, 1024           # 14 x 14 patch grid

# ---- data (raw COCO / VQAv2 files and downloaded images; not committed) ----
DATA_ROOT = Path(os.environ.get("PA3_T2_DATA", "data/task2"))
RAW = DATA_ROOT / "raw"
IMAGES = DATA_ROOT / "images"                  # images/<train2014|val2014>/COCO_<split>_<id:012d>.jpg

N_CAP_TRAIN_IMAGES, CAPS_PER_TRAIN_IMAGE, N_CAP_DEV = 6000, 3, 96
GROUND_DEV = dict(presence=12, counting=12)    # 24 pairs / 48 questions
GROUND_TEST = dict(presence=32, counting=32)   # 64 pairs / 128 questions
VQA_TRAIN = {"yes/no": 640, "number": 256, "other": 640}
VQA_DEV = {"yes/no": 12, "number": 8, "other": 12}
VQA_TEST = {"yes/no": 36, "number": 24, "other": 36}
VQA_MIN_AGREE = 8                              # of 10 annotators, for each image of a pair
# Annotation-derived grounding: label-quality filters (applied before any model is run)
PRESENCE_MIN_AREA = 0.01                       # positive: >= 1 non-crowd instance covering >= 1% of the image
COUNT_MIN_AREA = 0.005                         # counting: every instance must cover >= 0.5%, no crowd boxes
COUNT_RANGE = (1, 4)
POOL_PAIRS_PER_CATEGORY = 40                   # training pools: up to 40 pairs per category and question kind

# ---- prompts ----
CAPTION_PROMPT = "Write a short caption for this image."
SHORT_ANSWER_SUFFIX = "Answer the question using a single word or number."
CAPS = dict(question=80, caption=64, answer=16)  # content-token caps

# ---- training ----
MICRO_BS, GRAD_ACCUM = 8, 2
EFF_BS = MICRO_BS * GRAD_ACCUM                 # 16
WEIGHT_DECAY, GRAD_CLIP = 0.01, 1.0
BETAS, ADAM_EPS = (0.9, 0.999), 1e-8
WARMUP_FRAC, MIN_LR_FRAC = 0.05, 0.10
LORA = dict(r=16, alpha=32, dropout=0.05, targets=["q_proj", "k_proj", "v_proj", "o_proj"])
STAGES = [
    dict(name="caption", updates=2250, connector_lr=1e-3, lora_lr=None, mix={"caption": 1.0}),
    dict(name="instruct", updates=1500, connector_lr=1e-4, lora_lr=2e-5,
         mix={"presence": 0.50, "counting": 0.25, "caption": 0.25}),
    dict(name="vqa", updates=750, connector_lr=5e-5, lora_lr=1e-5,
         mix={"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}),
]
TOTAL_UPDATES = sum(s["updates"] for s in STAGES)  # 4500
STAGE_ENDS = [2250, 3750, 4500]
CKPT_EVERY = 250
DEV_EVAL_AT = sorted(set(range(0, TOTAL_UPDATES + 1, 500)) | set(STAGE_ENDS))
CALIB_IMAGES = 512                             # training images used for connector-scale calibration

# ---- evaluation ----
GEN_MAX_NEW, GEN_BS = 16, 8
CONDITIONS = ["correct", "complementary", "mismatched", "no_image"]
N_BOOT = 4000


def task_dir():
    """outputs/task2: shared data files at the top, one folder per training run under runs/."""
    return output_root() / "task2"


def data_dir():
    return task_dir() / "data"


def cache_dir():
    """Feature cache (gitignored); PA3_T2_CACHE moves it off the saved output folder on Kaggle."""
    return Path(os.environ["PA3_T2_CACHE"]) if os.environ.get("PA3_T2_CACHE") else task_dir() / "cache"


def run_dir(run="main"):
    return task_dir() / "runs" / run


def image_path(split, image_id):
    return IMAGES / split / f"COCO_{split}_{image_id:012d}.jpg"
