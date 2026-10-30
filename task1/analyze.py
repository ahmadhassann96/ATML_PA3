"""Collect Task 1 results into report tables and figures.

    python -m task1.analyze      -> outputs/task1/report/*.{csv,md,png}
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.utils import load_json, read_jsonl  # noqa: E402
from task1 import config as C  # noqa: E402

LABEL = {"baseline": "Baseline", "full_ft@625": "Full FT (mid)", "full_ft": "Full FT",
         "full_ft_l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
COLORS = {"full_ft": "#1f77b4", "full_ft_l2sp": "#ff7f0e", "lora4": "#2ca02c", "lora16": "#d62728",
          "oplora16": "#9467bd", "full_ft@625": "#7fb2d9", "baseline": "#7f7f7f"}
TASK_LABEL = {"gsm8k_cot": "GSM8K", "hellaswag": "HellaSwag", "boolq": "BoolQ", "arc_easy": "ARC-Easy"}


def out_dir():
    d = C.task_dir() / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ema(x, a=0.05):
    y, m = [], None
    for v in x:
        m = v if m is None else (1 - a) * m + a * v
        y.append(m)
    return np.array(y)


def loss_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    l2 = None
    for m in C.METHODS:
        p = C.run_dir(m) / "train_log.jsonl"
        if not p.exists():
            continue
        df = pd.DataFrame(read_jsonl(p))
        axes[0].plot(df["update"], df.sft_loss, color=COLORS[m], alpha=0.15, lw=0.6)
        axes[0].plot(df["update"], ema(df.sft_loss), color=COLORS[m], label=LABEL[m])
        axes[1].plot(df["update"], df.lr, color=COLORS[m], label=LABEL[m])
        if "omega" in df:
            l2 = df
    axes[0].set(xlabel="optimizer update", ylabel="SFT loss (token CE)", title="SFT loss (EMA 0.05; raw faint)")
    axes[0].legend()
    axes[1].set(xlabel="optimizer update", ylabel="learning rate", yscale="log", title="Learning rate")
    if l2 is not None:
        ax = axes[2]
        ax.plot(l2["update"], l2.lam_omega, label=r"$\lambda\Omega$", color="#ff7f0e")
        ax.plot(l2["update"], ema(l2.total_loss) - ema(l2.sft_loss), ls="--", color="k", lw=0.8,
                label="total - SFT (EMA)")
        ax.set(xlabel="optimizer update", ylabel=r"$\lambda\Omega$", title=r"L2-SP penalty ($\lambda=10^{-3}$)")
        ax2 = ax.twinx()
        ax2.plot(l2["update"], np.sqrt(l2.omega), color="#8c564b", alpha=0.6, label=r"$\|\theta-\theta_0\|_2$")
        ax2.set_ylabel(r"$\|\theta-\theta_0\|_2$")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir() / "loss_curves.png", dpi=150)
    plt.close(fig)


def benchmark_table():
    rows = []
    for cname, _, _ in C.CONDITIONS:
        p = C.task_dir() / "eval" / cname.replace("@", "_at_") / "summary.json"
        if not p.exists():
            continue
        s = load_json(p)
        r = dict(condition=cname)
        for t, sc in s["scores"].items():
            r[f"{t}:primary"] = sc.get(C.PRIMARY_METRIC[t])
            for k in ("acc,none", "acc_norm,none", "exact_match,strict-match", "exact_match,flexible-extract"):
                if k in sc:
                    r[f"{t}:{k.split(',')[0] if t != 'gsm8k_cot' else k.split(',')[1]}"] = sc[k]
        for k, v in s.get("gsm8k_diagnostics", {}).items():
            r[f"gsm8k:{k}"] = v
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("condition")
    if "baseline" in df.index:
        for t in C.PRIMARY_METRIC:
            col = f"{t}:primary"
            if col in df:
                df[f"{t}:delta_pp"] = 100 * (df[col] - df.loc["baseline", col])
    df.to_csv(out_dir() / "benchmarks.csv")

    # compact markdown table: primary score (%) and change from baseline (pp)
    lines = ["| Condition | " + " | ".join(TASK_LABEL[t] for t in C.PRIMARY_METRIC) + " |",
             "|---|" + "---|" * len(C.PRIMARY_METRIC)]
    for c in df.index:
        cells = []
        for t in C.PRIMARY_METRIC:
            v = df.loc[c].get(f"{t}:primary")
            d = df.loc[c].get(f"{t}:delta_pp")
            cells.append("—" if v is None or pd.isna(v) else
                         f"{100*v:.1f}" + ("" if c == "baseline" or d is None or pd.isna(d) else f" ({d:+.1f})"))
        lines.append(f"| {LABEL.get(c, c)} | " + " | ".join(cells) + " |")
    (out_dir() / "benchmarks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if "baseline" in df.index:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        conds = [c for c in df.index if c != "baseline"]
        w = 0.8 / max(len(conds), 1)
        tasks = [t for t in C.PRIMARY_METRIC if f"{t}:delta_pp" in df]
        for i, c in enumerate(conds):
            ax.bar(np.arange(len(tasks)) + i * w, [df.loc[c, f"{t}:delta_pp"] for t in tasks], w,
                   label=LABEL[c], color=COLORS[c])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(tasks)) + w * (len(conds) - 1) / 2, [TASK_LABEL[t] for t in tasks])
        ax.set_ylabel("change from baseline (pp)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir() / "benchmark_deltas.png", dpi=150)
        plt.close(fig)
    return df


def resources_table():
    rows = []
    for m in C.METHODS:
        p = C.run_dir(m) / "resources.json"
        if not p.exists():
            continue
        r = load_json(p)
        peak = r.get("peak_gpu_mem_allocated_bytes_max_over_sessions", r["peak_gpu_mem_allocated_bytes_this_session"])
        ov = r["method_overhead"]
        rows.append(dict(
            method=LABEL[m], trainable_params=r["trainable_params"], train_time_h=r["train_time_s"] / 3600,
            peak_gpu_mem_GB=peak / 1e9, gpu=r["gpu"],
            recovery_ckpt_GB=max(r["recovery_checkpoint_sizes"].values(), default=0) / 1e9,
            export_MB=max(r["export_sizes"].values(), default=0) / 1e6,
            skipped_updates=len(r["skipped_updates"]),
            displacement_final=r["displacement_l2"][max(r["displacement_l2"], key=int)] if r["displacement_l2"] else None,
            displacement_mid=r["displacement_l2"].get("625"),
            svd_time_s=ov.get("svd_time_s"), basis_MB=(ov.get("basis_bytes") or 0) / 1e6 or None,
            l2sp_ref_GB=(ov.get("l2sp_ref_bytes") or 0) / 1e9 or None,
            l2sp_penalty_time_s=ov.get("l2sp_penalty_time_s"),
            adapter_export_s=max(ov.get("adapter_export_time_s", {}).values(), default=None),
            protection_check=json.dumps(ov.get("protection_checks")) if ov.get("protection_checks") else None))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir() / "resources.csv", index=False)
        return df
    return None


def diagnostics_figure():
    p = C.task_dir() / "diagnostics" / "diagnostics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    order = [c for c in ["full_ft@625", "full_ft", "full_ft_l2sp", "lora4", "lora16", "oplora16"] if c in set(df.checkpoint)]
    metrics = [("rho_k", r"$\rho_k$ (protected output share)"), ("eta", r"$\eta=\|\Delta W\|_F/\|W_0\|_F$"),
               ("R_math_mean", r"mean $R(z)$, GSM8K probes"), ("R_nonmath_mean", r"mean $R(z)$, HellaSwag probes")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    w = 0.8 / len(order)
    for ax, (col, title) in zip(axes, metrics):
        for i, c in enumerate(order):
            sub = df[df.checkpoint == c].sort_values("layer")
            ax.bar(np.arange(len(C.DIAG_LAYERS)) + i * w, sub[col], w, label=LABEL[c], color=COLORS[c])
        ax.set_xticks(np.arange(len(C.DIAG_LAYERS)) + w * (len(order) - 1) / 2, [f"L{l}" for l in C.DIAG_LAYERS])
        ax.set_title(title, fontsize=10)
        if col != "rho_k":
            ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle("down_proj diagnostics (k = 128)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir() / "diagnostics.png", dpi=150)
    plt.close(fig)
    return df


def main():
    loss_curves()
    for name, fn in (("benchmarks", benchmark_table), ("resources", resources_table), ("diagnostics", diagnostics_figure)):
        df = fn()
        print(f"\n== {name} ==")
        print("(missing)" if df is None else df.to_string())
    print(f"\nwrote {out_dir()}")


if __name__ == "__main__":
    main()
