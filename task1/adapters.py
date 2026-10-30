"""LoRA and OPLoRA adapters written from scratch (no peft) so both share identical initial factors.

LoRA:    W = W0 + (alpha/r) B A,   A in R^{r x d_in} (random), B in R^{d_out x r} (zeros)
OPLoRA:  same factors, but the update uses projected factors
            B~ = B - U_k (U_k^T B)          (remove the protected output subspace)
            A~ = A - (A V_k) V_k^T          (remove the protected input subspace)
         where W0 = U S V^T and U_k, V_k are the k leading singular vectors (computed once in FP32).
         Only the low-rank factors are projected -- no dense d x d projection matrices.
"""
import math
import time

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, gen: torch.Generator):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        dev = base.weight.device
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        # kaiming_uniform_(a=sqrt(5)) as in the LoRA reference code -> U(-1/sqrt(d_in), 1/sqrt(d_in))
        bound = 1.0 / math.sqrt(d_in)
        a0 = (torch.rand(r, d_in, generator=gen, dtype=torch.float32) * 2 - 1) * bound
        self.lora_A = nn.Parameter(a0.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=dev))
        self.dropout = nn.Dropout(dropout)
        self.oplora = False

    def set_protection(self, Uk: torch.Tensor, Vk: torch.Tensor):
        """Uk: d_out x k, Vk: d_in x k (FP32, frozen)."""
        self.register_buffer("Uk", Uk.float().contiguous(), persistent=False)
        self.register_buffer("Vk", Vk.float().contiguous(), persistent=False)
        self.oplora = True

    def factors(self):
        """Effective (A, B) in FP32. For OPLoRA these are the projected factors."""
        A, B = self.lora_A, self.lora_B
        if self.oplora:
            with torch.autocast(device_type=A.device.type, enabled=False):
                B = B - self.Uk @ (self.Uk.T @ B)
                A = A - (A @ self.Vk) @ self.Vk.T
        return A, B

    def delta_weight(self):
        A, B = self.factors()
        return self.scaling * (B.float() @ A.float())

    def forward(self, x):
        y = self.base(x)
        A, B = self.factors()
        # factors are FP32 parameters; compute the low-rank path in the activation dtype (FP16)
        upd = (self.dropout(x) @ A.T.to(x.dtype)) @ B.T.to(x.dtype)
        return y + (upd * self.scaling).to(y.dtype)


def iter_targets(model, targets):
    for li, layer in enumerate(model.model.layers):
        for parent_name, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for t in targets:
                if hasattr(parent, t) and isinstance(getattr(parent, t), (nn.Linear, LoRALinear)):
                    yield f"model.layers.{li}.{parent_name}.{t}", parent, t


def inject(model, targets, r, alpha, dropout, seed, protect_k=None):
    """Replace target Linear layers with LoRALinear. Each module gets its own generator seeded by
    (seed, module index), so LoRA-16 and OPLoRA-16 start from identical unprojected factors."""
    overhead = dict(svd_time_s=0.0, basis_bytes=0)
    for idx, (name, parent, attr) in enumerate(iter_targets(model, targets)):
        base = getattr(parent, attr)
        gen = torch.Generator().manual_seed(seed * 1000 + idx)
        mod = LoRALinear(base, r, alpha, dropout, gen)
        if protect_k:
            t0 = time.perf_counter()
            W = base.weight.detach().float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            if W.is_cuda:
                torch.cuda.synchronize()
            overhead["svd_time_s"] += time.perf_counter() - t0
            # QR re-orthonormalizes the bases (same span) so the projection is exact to FP32 precision
            Uk = torch.linalg.qr(U[:, :protect_k])[0].contiguous()
            Vk = torch.linalg.qr(Vh[:protect_k].T)[0].contiguous()
            mod.set_protection(Uk, Vk)
            overhead["basis_bytes"] += (Uk.numel() + Vk.numel()) * 4
            del U, S, Vh, W
        setattr(parent, attr, mod)
    return overhead


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


@torch.no_grad()
def export_factors(model):
    """FP32 factors for every adapter. For OPLoRA we also store the projected factors that define
    the actual update (used for inference, merging and diagnostics)."""
    t0 = time.perf_counter()
    out = {}
    for name, m in lora_modules(model).items():
        rec = {"A": m.lora_A.detach().float().cpu(), "B": m.lora_B.detach().float().cpu(), "scaling": m.scaling}
        if m.oplora:
            A, B = m.factors()
            rec["A_proj"], rec["B_proj"] = A.detach().float().cpu(), B.detach().float().cpu()
        out[name] = rec
    return out, time.perf_counter() - t0


@torch.no_grad()
def check_protection(model):
    """Numerical check of the OPLoRA constraints: U_k^T dW = 0 and dW V_k = 0 (relative Frobenius)."""
    worst_out, worst_in = 0.0, 0.0
    for m in lora_modules(model).values():
        if not m.oplora:
            continue
        dW = m.delta_weight()
        n = dW.norm().item()
        if n == 0:
            continue
        worst_out = max(worst_out, (m.Uk.T @ dW).norm().item() / n)
        worst_in = max(worst_in, (dW @ m.Vk).norm().item() / n)
    return dict(max_rel_out_leak=worst_out, max_rel_in_leak=worst_in)


@torch.no_grad()
def merge_into(model, factors, use_projected=True):
    """Add (alpha/r) B A into the matching Linear weights of a *plain* (un-injected) model, in FP32."""
    mods = dict(model.named_modules())
    for name, rec in factors.items():
        lin = mods[name]
        A = rec["A_proj"] if (use_projected and "A_proj" in rec) else rec["A"]
        B = rec["B_proj"] if (use_projected and "B_proj" in rec) else rec["B"]
        dW = rec["scaling"] * (B.float() @ A.float())
        W = lin.weight.data
        lin.weight.data = (W.float() + dW.to(W.device)).to(W.dtype)
    return model
