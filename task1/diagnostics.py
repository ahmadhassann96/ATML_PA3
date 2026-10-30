"""Weight-space and activation diagnostics on down_proj of layers 0, 13, 27 (FP32 throughout).

For each checkpoint (five finals + Full FT midpoint) and layer, with W0 = U S V^T (k = 128):
  rho_k = ||U_k^T dW||_F^2 / ||dW||_F^2     subspace interference (output side)
  eta   = ||dW||_F / ||W0||_F               relative update magnitude
  R(z)  = ||dW z|| / ||W0 z||               activation correction on cached starting-model inputs z
where z is the input to down_proj at the last prompt token, cached once from the *starting* model on
the probe set. Reusing the same z isolates the local weight change (it ignores changes upstream).

    python -m task1.diagnostics            -> outputs/task1/diagnostics/{diagnostics.json,csv}
"""
import argparse

import numpy as np
import pandas as pd
import torch

from common.utils import save_json
from task1 import config as C
from task1 import probe
from task1.modeling import export_dir, load_base, load_factors, load_tokenizer

CHECKPOINTS = [("full_ft@625", "full_ft", 625), ("full_ft", "full_ft", 1250),
               ("full_ft_l2sp", "full_ft_l2sp", 1250), ("lora4", "lora4", 1250),
               ("lora16", "lora16", 1250), ("oplora16", "oplora16", 1250)]


@torch.no_grad()
def cache_activations(model, tok, items):
    store = {li: [] for li in C.DIAG_LAYERS}
    hooks = []
    for li in C.DIAG_LAYERS:
        def hook(_m, inp, li=li):
            store[li].append(inp[0][0, -1].detach().float().cpu())  # batch 1, no padding -> last token
        hooks.append(model.model.layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    for it in items:
        ids = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    return {li: torch.stack(v) for li, v in store.items()}  # (n_items, d_in)


def delta_w(name, method, update, li, W0):
    kind = C.METHODS[method]["kind"]
    if kind == "full":
        W = torch.load(export_dir(method, update) / "diag_down_proj_fp32.pt", map_location="cpu")[li]
        return W.float() - W0
    rec = load_factors(method, update)[f"model.layers.{li}.mlp.down_proj"]
    A = rec.get("A_proj", rec["A"]).float()
    B = rec.get("B_proj", rec["B"]).float()
    return rec["scaling"] * (B @ A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of checkpoint names")
    ap.add_argument("--smoke", action="store_true", help="use the tiny *_smoke training runs")
    args = ap.parse_args()
    out = C.task_dir() / ("diagnostics_smoke" if args.smoke else "diagnostics")
    import task1.modeling as M
    M.SMOKE = args.smoke
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    ps = probe.load()
    items = ps["items"]
    kinds = np.array([it["kind"] for it in items])
    model = load_base(torch.float32, "cuda").eval()
    Z = cache_activations(model, tok, items)
    W0s = {li: model.model.layers[li].mlp.down_proj.weight.detach().float().cpu() for li in C.DIAG_LAYERS}
    del model
    torch.cuda.empty_cache()
    torch.save(dict(Z=Z, kinds=kinds.tolist(), ids=[(it["set"], it["doc_id"]) for it in items]),
               out / "probe_activations.pt")

    bases = {}
    for li, W0 in W0s.items():
        U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        bases[li] = (U[:, :C.DIAG_K], Vh[:C.DIAG_K].T)

    rows, per_example = [], []
    for name, method, update in CHECKPOINTS:
        update = M.condition_spec(name)[1]
        if args.only and name not in args.only:
            continue
        try:
            for li in C.DIAG_LAYERS:
                W0, (Uk, Vk), z = W0s[li], bases[li], Z[li]
                dW = delta_w(name, method, update, li, W0)
                f2 = dW.pow(2).sum().item()
                rho_out = (Uk.T @ dW).pow(2).sum().item() / f2 if f2 > 0 else float("nan")
                rho_in = (dW @ Vk).pow(2).sum().item() / f2 if f2 > 0 else float("nan")
                eta = f2 ** 0.5 / W0.norm().item()
                R = ((z @ dW.T).norm(dim=1) / (z @ W0.T).norm(dim=1)).numpy()
                rows.append(dict(checkpoint=name, layer=li, rho_k=rho_out, rho_k_input_side=rho_in, eta=eta,
                                 dW_fro=f2 ** 0.5, R_math_mean=float(R[kinds == "math"].mean()),
                                 R_math_median=float(np.median(R[kinds == "math"])),
                                 R_nonmath_mean=float(R[kinds == "non_math"].mean()),
                                 R_nonmath_median=float(np.median(R[kinds == "non_math"])),
                                 R_ratio_math_over_nonmath=float(R[kinds == "math"].mean() / R[kinds == "non_math"].mean())))
                for it, r in zip(items, R):
                    per_example.append(dict(checkpoint=name, layer=li, set=it["set"], doc_id=it["doc_id"], R=float(r)))
                print(rows[-1])
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(per_example).to_csv(out / "activation_ratios_per_example.csv", index=False)
    save_json(dict(k=C.DIAG_K, layers=C.DIAG_LAYERS, rows=rows), out / "diagnostics.json")
    print(df.to_string())


if __name__ == "__main__":
    main()
