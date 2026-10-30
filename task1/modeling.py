"""Model loading for training, evaluation and analysis."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task1 import adapters
from task1 import config as C


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(C.MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(dtype=torch.float16, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(C.MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device)


SMOKE = False  # set by --smoke in eval/diagnostic scripts: read the tiny <method>_smoke runs instead
_SMOKE_UPDATE = {625: 2, 1250: 4}


def export_dir(method, update):
    if SMOKE:
        return C.run_dir(method + "_smoke") / "exports" / f"u{update}"
    return C.run_dir(method) / "exports" / f"u{update}"


def condition_spec(name):
    for cname, method, update in C.CONDITIONS:
        if cname == name:
            return method, (_SMOKE_UPDATE.get(update, update) if SMOKE else update)
    raise KeyError(f"unknown condition {name}; choose from {[c[0] for c in C.CONDITIONS]}")


def load_condition(name, device="cuda", dtype=torch.float16):
    """Plain (merged) eval model for one of the seven evaluation conditions, dropout disabled."""
    method, update = condition_spec(name)
    if method is None:
        model = load_base(dtype, device)
    elif C.METHODS[method]["kind"] == "full":
        d = export_dir(method, update) / "model"
        model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    else:
        model = load_merged_adapter(method, update, device=device, dtype=dtype)
    model.eval()
    return model


def load_factors(method, update):
    return torch.load(export_dir(method, update) / "adapters.pt", map_location="cpu")


def load_merged_adapter(method, update, device="cuda", dtype=torch.float16):
    """Fresh backbone copy + merged adapter update (projected factors for OPLoRA), merged in FP32."""
    model = load_base(torch.float32, device)
    adapters.merge_into(model, load_factors(method, update), use_projected=True)
    return model.to(dtype)


@torch.no_grad()
def merge_check(method, update, tok, device="cuda"):
    """Compare logits of the merged FP16 model with the unmerged (injected) FP16 model."""
    cfg = C.METHODS[method]
    factors = load_factors(method, update)
    texts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
        "The capital of France is",
        "A man is sitting on a roof. He",
        "Is the sky blue? Answer yes or no.",
    ]
    enc = tok(texts, return_tensors="pt", padding=True).to(device)

    unmerged = load_base(torch.float16, device)
    for p in unmerged.parameters():
        p.requires_grad_(False)
    adapters.inject(unmerged, C.LORA_TARGETS, cfg["r"], cfg["alpha"], 0.0, C.SEED,
                    protect_k=cfg.get("k") if cfg["kind"] == "oplora" else None)
    mods = adapters.lora_modules(unmerged)
    for name, rec in factors.items():
        mods[name].lora_A.data.copy_(rec["A"])
        mods[name].lora_B.data.copy_(rec["B"])
    unmerged.eval()
    lu = unmerged(**enc).logits.float()
    del unmerged
    torch.cuda.empty_cache()

    merged = load_merged_adapter(method, update, device)
    merged.eval()
    lm = merged(**enc).logits.float()
    del merged
    torch.cuda.empty_cache()

    m = enc["attention_mask"].bool()
    diff = (lu - lm).abs()[m]
    agree = (lu.argmax(-1) == lm.argmax(-1))[m].float().mean().item()
    return dict(max_abs_logit_diff=diff.max().item(), mean_abs_logit_diff=diff.mean().item(),
                argmax_agreement=agree, n_tokens=int(m.sum()))


def save_fp16_model(model, out_dir: Path):
    """Save an FP16 copy of an FP32-master model without touching the master weights."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.float16).cpu() for k, v in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)  # tied to embed_tokens; re-tied on load
    model.save_pretrained(out_dir, state_dict=sd, safe_serialization=True)
    model.config.save_pretrained(out_dir)
