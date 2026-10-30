"""Train one Task 1 variant: full_ft | full_ft_l2sp | lora4 | lora16 | oplora16.

    python -m task1.train --method lora16            # resumes automatically from the latest recovery checkpoint
    python -m task1.train --method full_ft --smoke   # tiny end-to-end check

Precision: FP16 autocast + dynamic GradScaler. Full FT keeps FP32 master weights; adapter runs use a
frozen FP16 backbone with FP32 adapters. Optimizer states are FP32. Gradients are unscaled before
clipping; overflowed (skipped) updates are logged.

Loss: next-token CE on assistant-response tokens + <|im_end|> only, summed over every microbatch and
divided by the number of supervised tokens in the whole effective batch (32 examples).
"""
import argparse
import functools
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup

from common.utils import (append_jsonl, dir_size_bytes, env_info, get_rng_state, save_json,
                          set_rng_state, set_seed)
from task1 import adapters
from task1 import config as C
from task1.data import SFTData
from task1.modeling import export_dir, load_base, load_tokenizer, save_fp16_model


# ----------------------------------------------------------------------------- loss
def _ce_chunk(lm_head, h, t):
    return F.cross_entropy(lm_head(h).float(), t, reduction="sum")


def sft_loss_sum(model, ids, att, lab):
    """Summed CE over supervised positions. lm_head + CE run on supervised positions only and in
    checkpointed chunks, so the full (T x 151k) logit tensor is never materialized."""
    h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    h, tgt = h[:, :-1], lab[:, 1:]
    m = tgt != -100
    h, tgt = h[m], tgt[m]
    fn = functools.partial(_ce_chunk, model.lm_head)
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, tgt.numel(), C.CE_CHUNK):
        total = total + checkpoint(fn, h[s : s + C.CE_CHUNK], tgt[s : s + C.CE_CHUNK], use_reentrant=False)
    return total


# ----------------------------------------------------------------------------- L2-SP
class L2SP:
    """Omega(theta) = sum_j ||theta_j - theta0_j||^2 over all unique parameters (tied counted once).
    Its gradient 2*lambda*(theta - theta0) is added to the unscaled SFT gradient before clipping."""

    def __init__(self, model, lam, ref_device):
        self.lam = lam
        self.params = [p for p in model.parameters() if p.requires_grad]  # parameters() dedups tied
        t0 = time.perf_counter()
        pin = ref_device == "cpu" and torch.cuda.is_available()
        self.ref = []
        for p in self.params:
            r = p.detach().clone().to(ref_device)
            self.ref.append(r.pin_memory() if pin else r)
        self.ref_device = ref_device
        self.setup_time_s = time.perf_counter() - t0
        self.ref_bytes = sum(r.numel() * r.element_size() for r in self.ref)
        self.transfer_time_s = 0.0  # cumulative time in apply() (includes CPU->GPU copies if ref on CPU)

    @torch.no_grad()
    def omega(self, add_grad=False):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        acc = torch.zeros((), dtype=torch.float64, device=self.params[0].device)
        for p, r in zip(self.params, self.ref):
            d = p.detach() - r.to(p.device, non_blocking=True)
            acc += (d * d).sum(dtype=torch.float64)
            if add_grad and p.grad is not None:
                p.grad.add_(d, alpha=2.0 * self.lam)
            del d
        val = acc.item()
        self.transfer_time_s += time.perf_counter() - t0
        return val


@torch.no_grad()
def displacement_from_pretrained(model):
    """||theta - theta0||_2 against a fresh FP32 copy of the starting checkpoint loaded on CPU."""
    ref = dict(load_base(torch.float32, "cpu").named_parameters())
    acc = 0.0
    for n, p in model.named_parameters():
        d = p.detach().float() - ref[n].to(p.device)
        acc += (d * d).sum(dtype=torch.float64).item()
    del ref
    return acc ** 0.5


# ----------------------------------------------------------------------------- setup
def build_model(method, smoke=False):
    cfg = C.METHODS[method]
    overhead = {}
    if cfg["kind"] == "full":
        model = load_base(torch.float32, "cuda")  # FP32 master weights
        for n, p in model.named_parameters():
            # smoke test only: train layers 0-1 so Full FT fits an 8 GB GPU; real runs train everything
            p.requires_grad_(not smoke or n.startswith(("model.layers.0.", "model.layers.1.")))
    else:
        model = load_base(torch.float16, "cuda")  # frozen FP16 backbone
        for p in model.parameters():
            p.requires_grad_(False)
        overhead = adapters.inject(
            model, C.LORA_TARGETS, cfg["r"], cfg["alpha"], cfg["dropout"], C.SEED,
            protect_k=cfg.get("k") if cfg["kind"] == "oplora" else None,
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, overhead


def param_groups(model, lr):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    return [
        {"params": decay, "weight_decay": C.WEIGHT_DECAY, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


def trainable_state(model, kind):
    if kind == "full":
        return {k: v for k, v in model.state_dict().items()}
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def ckpt_dir(rd: Path) -> Path:
    """Recovery checkpoints go to $PA3_CKPT/<run> if set (e.g. Kaggle scratch disk), else <run>/checkpoints."""
    root = os.environ.get("PA3_CKPT")
    return Path(root) / rd.name if root else rd / "checkpoints"


def latest_checkpoint(rd: Path):
    ck = sorted(ckpt_dir(rd).glob("u*.pt"), key=lambda p: int(p.stem[1:]))
    return ck[-1] if ck else None


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(C.METHODS))
    ap.add_argument("--smoke", action="store_true", help="4 tiny updates, writes to <run>_smoke")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--keep_all_recovery", action="store_true",
                    help="keep every recovery checkpoint (default keeps only the latest to save disk)")
    ap.add_argument("--l2sp_ref_device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--stop_after", type=int, default=None,
                    help="exit cleanly after this many updates (e.g. to fit a Kaggle session)")
    args = ap.parse_args()

    method, cfg = args.method, C.METHODS[args.method]
    micro, accum = C.MICRO_BS, C.GRAD_ACCUM
    total, recovery, evals = C.TOTAL_UPDATES, C.RECOVERY_UPDATES, C.EVAL_UPDATES[method]
    rd = C.run_dir(method)
    if args.smoke:
        accum, total = 2, 4
        recovery, evals = [2, 4], ([2, 4] if method == "full_ft" else [4])
        rd = rd.with_name(method + "_smoke")
    eff = micro * accum
    rd.mkdir(parents=True, exist_ok=True)

    set_seed(C.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = SFTData(limit=total * eff)
    assert len(data) >= total * eff, "run `python -m task1.data` first"
    tok = load_tokenizer()

    torch.cuda.reset_peak_memory_stats()
    model, overhead = build_model(method, smoke=args.smoke)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(param_groups(model, cfg["lr"]), lr=cfg["lr"], betas=C.BETAS, eps=C.ADAM_EPS,
                            fused=True)
    sched = get_cosine_schedule_with_warmup(opt, C.WARMUP_UPDATES, total)
    scaler = torch.amp.GradScaler("cuda")

    l2sp = None
    if cfg["kind"] == "full" and cfg["l2sp_lambda"] > 0:
        dev = args.l2sp_ref_device
        if dev == "auto":  # keep theta0 on GPU only if it comfortably fits
            dev = "cuda" if torch.cuda.get_device_properties(0).total_memory > 20e9 else "cpu"
        l2sp = L2SP(model, cfg["l2sp_lambda"], dev)

    state = dict(update=0, train_time_s=0.0, ckpt_time_s=0.0, export_time_s=0.0, skipped=[],
                 recovery_sizes={}, export_sizes={}, displacement={}, protection_checks={},
                 adapter_export_time_s={}, sessions=0)
    log_path = rd / "train_log.jsonl"
    ck = None if args.no_resume else latest_checkpoint(rd)
    if ck is not None:
        print(f"resuming from {ck}")
        s = torch.load(ck, map_location="cpu", weights_only=False)
        if cfg["kind"] == "full":
            model.load_state_dict(s["trainable"])
        else:
            named = dict(model.named_parameters())
            for n, v in s["trainable"].items():
                named[n].data.copy_(v)
        opt.load_state_dict(s["optimizer"])
        sched.load_state_dict(s["scheduler"])
        scaler.load_state_dict(s["scaler"])
        set_rng_state(s["rng"])
        state = s["state"]
        if l2sp is not None:
            l2sp.transfer_time_s = state.get("l2sp_time_s", 0.0)
        # drop log lines written after this checkpoint (crash between checkpoint and exit)
        if log_path.exists():
            lines = [l for l in log_path.read_text(encoding="utf-8").splitlines()
                     if l.strip() and json.loads(l)["update"] <= state["update"]]
            log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        del s
    elif log_path.exists():
        log_path.unlink()
    state["sessions"] += 1

    save_json(dict(method=method, method_cfg=cfg, micro_bs=micro, grad_accum=accum, eff_bs=eff,
                   total_updates=total, warmup=C.WARMUP_UPDATES, betas=C.BETAS, eps=C.ADAM_EPS,
                   weight_decay=C.WEIGHT_DECAY, grad_clip=C.GRAD_CLIP, max_len=C.MAX_LEN, seed=C.SEED,
                   lora_targets=C.LORA_TARGETS if cfg["kind"] != "full" else None,
                   recovery_updates=recovery, eval_updates=evals, smoke=args.smoke,
                   l2sp_ref_device=l2sp.ref_device if l2sp else None, env=env_info()),
              rd / "config.json")

    pad = tok.pad_token_id
    session_start = state["update"]
    for u in range(state["update"], total):
        t_upd = time.perf_counter()
        base = u * eff
        batches = [data.microbatch(range(base + m * micro, base + (m + 1) * micro), pad) for m in range(accum)]
        n_tok = sum(int((lab[:, 1:] != -100).sum()) for _, _, lab in batches)
        loss_sum = 0.0
        for ids, att, lab in batches:
            ids, att, lab = ids.cuda(non_blocking=True), att.cuda(non_blocking=True), lab.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                ls = sft_loss_sum(model, ids, att, lab)
            scaler.scale(ls / n_tok).backward()
            loss_sum += ls.item()
        scaler.unscale_(opt)
        omega = l2sp.omega(add_grad=True) if l2sp else None
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, C.GRAD_CLIP).item()
        lr = sched.get_last_lr()[0]
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        opt.zero_grad(set_to_none=True)
        sched.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_upd
        state["train_time_s"] += dt
        state["update"] = u + 1
        if skipped:
            state["skipped"].append(u + 1)

        sft = loss_sum / n_tok
        rec = dict(update=u + 1, examples_seen=(u + 1) * eff, lr=lr, sft_loss=sft, n_tokens=n_tok,
                   grad_norm=gnorm, loss_scale=scale_before, skipped=skipped, time_s=dt)
        if l2sp:
            rec.update(omega=omega, lam_omega=l2sp.lam * omega, total_loss=sft + l2sp.lam * omega)
        append_jsonl(rec, log_path)
        if (u + 1) % 10 == 0 or u == session_start:
            print(f"[{method}] upd {u+1}/{total} loss {sft:.4f} lr {lr:.2e} gnorm {gnorm:.3f} "
                  f"{dt:.1f}s/upd" + (f" omega {omega:.3e}" if l2sp else "") + (" SKIPPED" if skipped else ""),
                  flush=True)

        if (u + 1) in evals:
            t0 = time.perf_counter()
            ed = export_dir(method, u + 1) if not args.smoke else rd / "exports" / f"u{u+1}"
            ed.mkdir(parents=True, exist_ok=True)
            if cfg["kind"] == "full":
                save_fp16_model(model, ed / "model")
                # FP32 copies of the diagnostic down_proj weights (FP16 rounding would swamp small updates)
                torch.save({li: model.model.layers[li].mlp.down_proj.weight.detach().float().cpu()
                            for li in C.DIAG_LAYERS}, ed / "diag_down_proj_fp32.pt")
                disp = (l2sp.omega() ** 0.5) if l2sp else displacement_from_pretrained(model)
                state["displacement"][str(u + 1)] = disp
                print(f"  ||theta - theta0||_2 = {disp:.4f}")
            else:
                factors, t_exp = adapters.export_factors(model)
                torch.save(factors, ed / "adapters.pt")
                state["adapter_export_time_s"][str(u + 1)] = t_exp
                if cfg["kind"] == "oplora":
                    state["protection_checks"][str(u + 1)] = adapters.check_protection(model)
                    print("  OPLoRA constraint check:", state["protection_checks"][str(u + 1)])
            state["export_sizes"][str(u + 1)] = dir_size_bytes(ed)
            state["export_time_s"] += time.perf_counter() - t0

        if (u + 1) in recovery:
            t0 = time.perf_counter()
            if l2sp:
                state["l2sp_time_s"] = l2sp.transfer_time_s
            ckd = ckpt_dir(rd)
            ckd.mkdir(parents=True, exist_ok=True)
            path = ckd / f"u{u+1}.pt"
            torch.save(dict(trainable=trainable_state(model, cfg["kind"]), optimizer=opt.state_dict(),
                            scheduler=sched.state_dict(), scaler=scaler.state_dict(), rng=get_rng_state(),
                            state=state, data_position=(u + 1) * eff), str(path) + ".tmp")
            Path(str(path) + ".tmp").replace(path)
            state["recovery_sizes"][str(u + 1)] = path.stat().st_size
            if not args.keep_all_recovery:
                for old in ckd.glob("u*.pt"):
                    if old != path:
                        old.unlink()
            state["ckpt_time_s"] += time.perf_counter() - t0
            print(f"  saved recovery checkpoint {path.name} ({path.stat().st_size/1e9:.2f} GB)")

        if args.stop_after is not None and (u + 1) - session_start >= args.stop_after:
            print("stop_after reached; resume later with the same command")
            break

    res = dict(
        method=method, updates_done=state["update"], total_updates=total,
        trainable_params=n_trainable, total_params_in_memory=n_total,
        train_time_s=state["train_time_s"], checkpoint_time_s=state["ckpt_time_s"],
        export_time_s=state["export_time_s"], sessions=state["sessions"],
        peak_gpu_mem_allocated_bytes_this_session=torch.cuda.max_memory_allocated(),
        peak_gpu_mem_reserved_bytes_this_session=torch.cuda.max_memory_reserved(),
        skipped_updates=state["skipped"], recovery_checkpoint_sizes=state["recovery_sizes"],
        export_sizes=state["export_sizes"], displacement_l2=state["displacement"],
        method_overhead=dict(**overhead,
                             adapter_export_time_s=state["adapter_export_time_s"],
                             protection_checks=state["protection_checks"],
                             l2sp_ref_bytes=l2sp.ref_bytes if l2sp else None,
                             l2sp_ref_device=l2sp.ref_device if l2sp else None,
                             l2sp_ref_setup_s=l2sp.setup_time_s if l2sp else None,
                             l2sp_penalty_time_s=l2sp.transfer_time_s if l2sp else None),
        gpu=torch.cuda.get_device_name(0),
    )
    prev = rd / "resources.json"
    if prev.exists():  # keep the max peak memory across resumed sessions
        old = json.loads(prev.read_text(encoding="utf-8"))
        for k in ("peak_gpu_mem_allocated_bytes_this_session", "peak_gpu_mem_reserved_bytes_this_session"):
            res[k.replace("_this_session", "_max_over_sessions")] = max(
                res[k], old.get(k.replace("_this_session", "_max_over_sessions"), 0), old.get(k, 0))
    save_json(res, rd / "resources.json")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
