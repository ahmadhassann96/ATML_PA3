# ATML PA3 — Fine-tuning, VLM grounding, test-time compute

Code for CS-6304 Programming Assignment 3. Each task is a separate package (`task1/`, `task2/`, `task3/`);
`common/` holds shared helpers. All results are written under `$PA3_OUT` (default `./outputs`).
Small machine-readable results (JSON/CSV/PNG) are committed; datasets and large checkpoints are not.

## Environment

```bash
uv venv .venv --python 3.11          # or python -m venv .venv
.venv\Scripts\activate               # Windows   (source .venv/bin/activate on Linux)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs CUDA >= 12.8
pip install -r requirements.txt
```

Tested locally on Windows 11, RTX 5060 Laptop (8 GB), Python 3.11, torch 2.11+cu128. Full fine-tuning runs
need ~12 GB and were run on Kaggle (T4 16 GB) with `notebooks/task1_kaggle_full_ft.ipynb`.
Exact library versions for every run are stored in each run's `config.json` (`env` field).

## Task 1 — LLM fine-tuning (Qwen3-0.6B on MetaMathQA)

| Step | Command | Output |
|---|---|---|
| Download model + all datasets once | `python -m task1.prefetch` (then optionally `set HF_HUB_OFFLINE=1`) | HF cache |
| Build training set (first 40k eligible MetaMathQA rows) | `python -m task1.data` | `outputs/task1/data/selection.json` |
| Train a variant | `python -m task1.train --method {full_ft,full_ft_l2sp,lora4,lora16,oplora16}` | `outputs/task1/runs/<method>/` |
| Evaluate a condition | `python -m task1.evaluate --condition {baseline,full_ft@625,full_ft,full_ft_l2sp,lora4,lora16,oplora16}` | `outputs/task1/eval/<condition>/` |
| Probe set (32 GSM8K + 32 HellaSwag, seed 6304) | `python -m task1.probe` | `outputs/task1/probe_set.json` |
| Weight / activation diagnostics | `python -m task1.diagnostics` | `outputs/task1/diagnostics/` |
| Qualitative generations (per condition, then collect) | `python -m task1.qualitative --yaml first_plot_questions.yaml --condition <c>` then `--collect` | `outputs/task1/qualitative/` |
| Tables and figures | `python -m task1.analyze` | `outputs/task1/report/` |
| Everything after training, in order (skips finished steps) | `powershell -ExecutionPolicy Bypass -File task1\run_remaining.ps1` | all of the above |
| Evaluate some conditions on Kaggle (2× T4 in parallel) instead | `notebooks/task1_kaggle_eval.ipynb`; locally add `-Skip <conditions>` to the line above | `eval_results.zip` → `outputs/task1/eval/` |

Quick end-to-end check (4 tiny updates, separate `<method>_smoke` folder): `python -m task1.train --method lora16 --smoke`,
then `python -m task1.evaluate --condition lora16 --smoke --limit 8` (`--smoke` also works for `diagnostics` and `qualitative`).
For Full FT the smoke test trains only layers 0–1 so it fits an 8 GB GPU.

Measured on the RTX 5060 Laptop: LoRA-16 ≈ 6.7 s/update (≈ 2.3 h per run, 3.4 GB peak);
one full evaluation condition ≈ 1 h (GSM8K generation dominates).

`task1/__init__.py` disables the Hugging Face Xet download client and sets request timeouts — Xet downloads
hung indefinitely on our Windows machine.

Training resumes automatically from the latest recovery checkpoint (updates 312/625/937/1250) when re-run.
Set `PA3_CKPT=<dir>` to write recovery checkpoints somewhere other than the run folder.

**Implementation notes**

- *Loss*: CE on assistant-response tokens and `<|im_end|>` only; per-microbatch summed CE divided by the
  number of supervised tokens in the full effective batch of 32. `lm_head` + CE are computed only at
  supervised positions in checkpointed chunks (`task1/train.py: sft_loss_sum`) to avoid materializing
  full-vocabulary logits; this changes memory, not the objective.
- *L2-SP*: Ω = Σ‖θ − θ₀‖² over all unique parameters (tied embedding counted once), not averaged.
  Its gradient 2λ(θ − θ₀) is added to the unscaled SFT gradient before clipping. θ₀ lives on the GPU
  if it has > 20 GB, otherwise in pinned CPU memory; storage and time spent are logged.
- *LoRA / OPLoRA*: implemented from scratch in `task1/adapters.py` (no `peft`) so LoRA-16 and OPLoRA-16
  share identical initial factors (per-module generators seeded from 6304). OPLoRA projects the *factors*
  (B̃ = B − U_k U_kᵀB, Ã = A − A V_k V_kᵀ) with bases from an FP32 SVD of each starting weight; the
  constraint leak ‖U_kᵀΔW‖/‖ΔW‖ and ‖ΔW V_k‖/‖ΔW‖ is checked at export.
- *Evaluation*: lm-evaluation-harness with an `HFLM` subclass that always passes `enable_thinking=False`
  to the chat template. Adapters are merged into a fresh FP32 backbone, then cast to FP16; merged vs.
  unmerged logits are compared and stored in `summary.json`. GSM8K truncation is flagged when the
  re-tokenized response is within 8 tokens of the 1,024 cap (lm-eval strips stop strings).

## Task 2 — VLM and visual grounding

_TODO_

## Task 3 — Test-time compute

_TODO_

## Attribution

- LoRA initialization follows the reference LoRA / `peft` convention (Kaiming-uniform A, zero B); the code is our own.
- Benchmarks, prompts, filters and metrics: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- Models and datasets: Qwen3-0.6B, MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy (Hugging Face Hub).
- Code was written with assistance from Claude (Anthropic); all of it was reviewed by the author.
