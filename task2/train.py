"""Train the SigLIP-Qwen VLM through the three-stage curriculum (one continuous run, resumable).

    python -m task2.train                       # main run, 196 visual tokens -> outputs/task2/runs/main
    python -m task2.train --pool 2 --run pool49 # optional 49-token run
    python -m task2.train --smoke               # a few updates per stage on a tiny dev subset

Stage 1 (2,250 updates): connector only, two shuffled passes over the 18,000 caption examples.
Stage 2 (1,500): connector + LoRA (q/k/v/o, r=16, alpha=32, dropout 0.05); draws: presence 50%,
                 counting 25%, caption replay 25%.
Stage 3 (750):   same parameters; draws: VQAv2 70%, presence / counting / caption replay 10% each.
A draw adds a complete question pair (two examples) or two caption examples; 8 draws = one update of 16.
Each stage has its own AdamW optimizer and schedule (5% linear warmup, cosine decay to 10% of peak).
FP16 autocast + dynamic gradient scaling over a frozen FP16 Qwen; connector, adapters and optimizer
states are FP32; gradients are unscaled before clipping to global norm 1.0.
Loss: token CE averaged within each example, then across the 16 examples of the effective batch.
"""
import argparse
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from common.utils import append_jsonl, env_info, get_rng_state, load_json, read_jsonl, save_json, set_rng_state, set_seed
from task2 import config as C
from task2 import model as M
from task2.features import FeatureStore


# ----------------------------------------------------------------------------- data + sampler
class Data:
    def __init__(self, smoke=False):
        d = C.data_dir()
        r = lambda n: read_jsonl(d / f"{n}.jsonl")  # noqa: E731
        self.cap_train, self.cap_dev = r("captions_train"), r("captions_dev")
        self.vqa_train, self.vqa_dev, self.ground_dev = r("vqa_train"), r("vqa_dev"), r("ground_dev")
        self.pools = dict(presence=r("ground_pool_presence"), counting=r("ground_pool_counting"), vqa=self.vqa_train)
        if smoke:
            self.cap_dev, self.vqa_dev, self.ground_dev = self.cap_dev[:8], self.vqa_dev[:4], self.ground_dev[:4]


def caption_example(c, comp="caption"):
    return dict(question=C.CAPTION_PROMPT, target=c["caption"], kind="caption", split=c["split"],
                image_id=c["image_id"], comp=comp, suffix=False)


def pair_examples(p, comp):
    return [dict(question=p["question"], target=it["target"], kind="answer", split=it["split"],
                 image_id=it["image_id"], comp=comp, suffix=True) for it in p["items"]]


class Sampler:
    def __init__(self, data, si, stage):
        self.data, self.si, self.stage = data, si, stage
        if si == 0:
            g = np.random.default_rng([C.SEED, 1000])
            n = len(data.cap_train)
            passes = math.ceil(stage["updates"] * C.EFF_BS / n)
            self.order = np.concatenate([g.permutation(n) for _ in range(passes)])
        self.rng = np.random.default_rng([C.SEED, 1000 + si])
        self.comps = list(stage["mix"])
        self.probs = np.array([stage["mix"][k] for k in self.comps])

    def update(self, k):
        """Examples for the k-th update of this stage (16 examples, 8 draws) and the drawn components."""
        if self.si == 0:
            idx = self.order[k * C.EFF_BS:(k + 1) * C.EFF_BS]
            return [caption_example(self.data.cap_train[i]) for i in idx], ["caption"] * (C.EFF_BS // 2)
        ex, comps = [], []
        for _ in range(C.EFF_BS // 2):
            comp = self.comps[self.rng.choice(len(self.comps), p=self.probs)]
            comps.append(comp)
            if comp == "caption":
                ii = self.rng.integers(len(self.data.cap_train), size=2)
                ex += [caption_example(self.data.cap_train[i], "caption_replay") for i in ii]
            else:
                pool = self.data.pools[comp]
                ex += pair_examples(pool[self.rng.integers(len(pool))], comp)
        return ex, comps

    def state(self):
        return self.rng.bit_generator.state

    def load_state(self, s):
        self.rng.bit_generator.state = s


def make_batch(prompter, store, examples, image=True, left=False, drop_termination=False):
    encs = [prompter.encode(e["question"], image, e.get("target"), e.get("kind", "answer"), e.get("suffix", True))
            for e in examples]
    feats = [store.get(e["split"], e["image_id"]) if image else None for e in examples]
    return M.collate(encs, feats, prompter.placeholder, left=left, drop_termination=drop_termination)


# ----------------------------------------------------------------------------- development evaluation
@torch.no_grad()
def batched_losses(lm, conn, prompter, store, examples, image_of, drop_termination, bs=16):
    """image_of(example) -> (split, image_id) or None. Returns per-example losses (list)."""
    out = []
    for s in range(0, len(examples), bs):
        chunk = examples[s:s + bs]
        imgs = [image_of(e) for e in chunk]
        groups = defaultdict(list)  # with / without image are batched separately (prefix lengths differ)
        for i, im in enumerate(imgs):
            groups[im is not None].append(i)
        res = [None] * len(chunk)
        for has_img, ii in groups.items():
            encs = [prompter.encode(chunk[i]["question"], has_img, chunk[i]["target"], chunk[i]["kind"],
                                    chunk[i]["suffix"]) for i in ii]
            feats = [store.get(*imgs[i]) if has_img else None for i in ii]
            b = M.to_device(M.collate(encs, feats, prompter.placeholder, drop_termination=drop_termination))
            with torch.autocast("cuda", dtype=torch.float16):
                l = M.example_losses(lm, conn, b)
            for i, v in zip(ii, l.tolist()):
                res[i] = v
        out += res
    return out


def pair_items(pairs):
    """Flatten pairs into examples with partner info."""
    ex = []
    for p in pairs:
        for j, it in enumerate(p["items"]):
            partner = p["items"][1 - j]
            ex.append(dict(question=p["question"], target=it["target"], kind="answer", suffix=True,
                           split=it["split"], image_id=it["image_id"], pair_id=p["pair_id"], qtype=p["kind"],
                           partner=(partner["split"], partner["image_id"])))
    return ex


@torch.no_grad()
def dev_eval(lm, conn, prompter, store, data, update):
    was = lm.training
    lm.eval(), conn.eval()
    t0 = M.now()
    rec = dict(update=update)
    caps = [caption_example(c) for c in data.cap_dev]
    rec["caption_loss"] = float(np.mean(batched_losses(lm, conn, prompter, store, caps,
                                                       lambda e: (e["split"], e["image_id"]), False)))
    for suite, pairs in (("grounding", data.ground_dev), ("vqa", data.vqa_dev)):
        ex = pair_items(pairs)
        nll = {}
        for cond, f in (("correct", lambda e: (e["split"], e["image_id"])), ("complementary", lambda e: e["partner"]),
                        ("no_image", lambda e: None)):
            nll[cond] = np.array(batched_losses(lm, conn, prompter, store, ex, f, True))
            rec[f"{suite}_nll_{cond}"] = float(nll[cond].mean())
        rec[f"{suite}_swap_penalty"] = float((nll["complementary"] - nll["correct"]).mean())
        for qt in sorted({e["qtype"] for e in ex}):
            m = np.array([e["qtype"] == qt for e in ex])
            rec[f"{suite}/{qt}_nll_correct"] = float(nll["correct"][m].mean())
            rec[f"{suite}/{qt}_swap_penalty"] = float((nll["complementary"][m] - nll["correct"][m]).mean())
    rec["time_s"] = M.now() - t0
    if was:
        lm.train(), conn.train()
    return rec


# ----------------------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate(conn, lm, prompter, store, data):
    """Fix s so that the mean per-token L2 norm of the initial visual tokens equals that of Qwen's input
    embeddings for the text tokens of the same training caption examples (training data only)."""
    g = np.random.default_rng([C.SEED, 2000])
    first = {}  # one caption example per distinct training image
    for c in data.cap_train:
        first.setdefault(c["image_id"], c)
    firsts = list(first.values())
    n = min(C.CALIB_IMAGES, len(firsts))
    ex = [caption_example(firsts[i]) for i in sorted(g.choice(len(firsts), size=n, replace=False))]
    conn.scale.fill_(1.0)
    vis = []
    for s in range(0, len(ex), 64):
        f = torch.stack([store.get(e["split"], e["image_id"]) for e in ex[s:s + 64]]).cuda().float()
        vis.append(conn(f).norm(dim=-1).flatten())
    vis_norm = torch.cat(vis).mean().item()
    ids = []
    for e in ex:
        enc = prompter.encode(e["question"], False, e["target"], "caption", False)
        ids += enc["ids"]
    emb = lm.get_input_embeddings().weight
    text_norm = emb[torch.tensor(ids, device=emb.device)].float().norm(dim=-1).mean().item()
    scale = text_norm / vis_norm
    conn.scale.fill_(scale)
    return dict(rule="s = mean ||e_text|| / mean ||g_phi(F)|| at initialization, on one caption example for each of "
                     f"{n} training images "
                     "(text: their prompt + caption tokens; visual: their 196 tokens)",
                n_examples=len(ex), n_text_tokens=len(ids), text_token_norm=text_norm,
                visual_token_norm_before=vis_norm, scale=scale)


# ----------------------------------------------------------------------------- optimizer
def lr_lambda(U):
    w = max(1, math.ceil(C.WARMUP_FRAC * U))

    def f(t):
        if t < w:
            return (t + 1) / w
        prog = min(1.0, (t - w) / max(1, U - w))
        return C.MIN_LR_FRAC + (1 - C.MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def make_optimizer(conn, lm, stage, U):
    groups = [dict(params=[conn.fc1.weight, conn.fc2.weight], lr=stage["connector_lr"], weight_decay=C.WEIGHT_DECAY),
              dict(params=[conn.fc1.bias, conn.fc2.bias], lr=stage["connector_lr"], weight_decay=0.0)]
    lora = [p for n, p in lm.named_parameters() if p.requires_grad]
    if lora:
        groups.append(dict(params=lora, lr=stage["lora_lr"], weight_decay=C.WEIGHT_DECAY))
    opt = torch.optim.AdamW(groups, betas=C.BETAS, eps=C.ADAM_EPS)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(U))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--pool", type=int, default=1, choices=[1, 2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop_after", type=int, default=None, help="exit after this many updates (resume later)")
    a = ap.parse_args()
    run = a.run + ("_smoke" if a.smoke else "")
    rd = C.run_dir(run)
    rd.mkdir(parents=True, exist_ok=True)
    stages = [dict(s) for s in C.STAGES]
    ckpt_every, dev_at = C.CKPT_EVERY, set(C.DEV_EVAL_AT)
    if a.smoke:
        for s, n in zip(stages, (30, 6, 6)):
            s["updates"] = n
        ckpt_every, dev_at = 10, {0, 30, 36, 42}
    ends = [int(x) for x in np.cumsum([s["updates"] for s in stages])]
    total = ends[-1]

    set_seed(C.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = Data(a.smoke)
    store = FeatureStore(a.pool)
    tok = M.load_tokenizer()
    prompter = M.Prompter(tok, store.n_tokens)
    lm = M.load_lm()
    lm.config.use_cache = False
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(C.SEED)
    conn = M.Connector().cuda()
    log_path, dev_path = rd / "train_log.jsonl", rd / "dev_log.jsonl"

    ckd = rd / "checkpoints"
    cks = sorted(ckd.glob("u*.pt"), key=lambda p: int(p.stem[1:])) if ckd.exists() else []
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False) if cks else None
    if ck is not None:
        state = ck["state"]
        print(f"resuming from {cks[-1].name}")
        if state["lora"]:
            M.add_lora(lm)
        M.load_trainable(conn, lm, {k: v.cuda() for k, v in ck["trainable"].items()})
        set_rng_state(ck["rng"])
        for p in (log_path, dev_path):  # drop lines written after the checkpoint
            if p.exists():
                keep = [l for l in p.read_text(encoding="utf-8").splitlines()
                        if l.strip() and json.loads(l)["update"] <= state["update"]]
                p.write_text("".join(k + "\n" for k in keep), encoding="utf-8")
    else:
        for p in (log_path, dev_path):
            if p.exists():
                p.unlink()
        state = dict(update=0, lora=False, sessions=0, mixture={}, examples={}, stage_res={}, dev_done=[],
                     skipped=[], checks={}, dev_time_s=0.0, ckpt_time_s=0.0)
        state["calibration"] = calibrate(conn, lm, prompter, store, data)
        print("calibration:", state["calibration"])
        f = store.get(data.cap_train[0]["split"], data.cap_train[0]["image_id"])
        state["checks"]["feature_shape"] = list(f.shape)
        assert tuple(f.shape) == (store.n_tokens, C.D_VIS)
        dc = load_json(C.data_dir() / "data_checks.json")
        dev_ims = {(it["split"], it["image_id"]) for p in data.vqa_dev + data.ground_dev for it in p["items"]} | \
                  {(c["split"], c["image_id"]) for c in data.cap_dev}
        train_ims = {(c["split"], c["image_id"]) for c in data.cap_train} | \
                    {(it["split"], it["image_id"]) for k in ("presence", "counting", "vqa") for p in data.pools[k]
                     for it in p["items"]}
        state["checks"]["split_isolation"] = dict(data_checks=dc["isolation"],
                                                  dev_images_in_training=len(dev_ims & train_ims))
        assert not (dev_ims & train_ims)
        state["checks"]["prompt_template"] = prompter.template_text
        state["checks"]["example_prompts"] = {
            k: tok.decode(prompter.encode(e["question"], True, e["target"], e["kind"], e["suffix"])["ids"])
            .replace(tok.decode([prompter.placeholder]) * store.n_tokens, f"<{store.n_tokens} visual tokens>")
            for k, e in (("caption", caption_example(data.cap_train[0])),
                         ("presence", pair_examples(data.pools["presence"][0], "presence")[0]),
                         ("vqa", pair_examples(data.vqa_train[0], "vqa")[0]))}
    state["sessions"] += 1
    save_json(dict(run=run, pool=a.pool, n_visual_tokens=store.n_tokens, stages=stages, micro_bs=C.MICRO_BS,
                   grad_accum=C.GRAD_ACCUM, eff_bs=C.EFF_BS, weight_decay=C.WEIGHT_DECAY, grad_clip=C.GRAD_CLIP,
                   betas=C.BETAS, eps=C.ADAM_EPS, warmup_frac=C.WARMUP_FRAC, min_lr_frac=C.MIN_LR_FRAC,
                   lora=C.LORA, caps=C.CAPS, caption_prompt=C.CAPTION_PROMPT, answer_suffix=C.SHORT_ANSWER_SUFFIX,
                   seed=C.SEED, ckpt_every=ckpt_every, dev_eval_at=sorted(dev_at), lm=C.LM_ID, vision=C.VISION_ID,
                   env=env_info()), rd / "config.json")

    def save_ckpt(opt, sched, scaler, sampler):
        t0 = time.perf_counter()
        ckd.mkdir(parents=True, exist_ok=True)
        p = ckd / f"u{state['update']}.pt"
        torch.save(dict(trainable={k: v.cpu() for k, v in M.trainable_state(conn, lm).items()},
                        optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                        sampler=sampler.state(), rng=get_rng_state(), state=state), str(p) + ".tmp")
        (ckd / (p.name + ".tmp")).replace(p)
        for old in ckd.glob("u*.pt"):
            if old != p:
                old.unlink()
        state["ckpt_time_s"] += time.perf_counter() - t0

    def run_dev(u):
        if u in state["dev_done"]:
            return
        r = dev_eval(lm, conn, prompter, store, data, u)
        r["stage"] = int(np.searchsorted(ends, u, side="left")) + 1 if u > 0 else 0
        append_jsonl(r, dev_path)
        state["dev_done"].append(u)
        state["dev_time_s"] += r["time_s"]
        print(f"  dev@{u}: cap {r['caption_loss']:.3f} | ground nll {r['grounding_nll_correct']:.3f} "
              f"swap {r['grounding_swap_penalty']:+.3f} | vqa nll {r['vqa_nll_correct']:.3f} "
              f"swap {r['vqa_swap_penalty']:+.3f} ({r['time_s']:.0f}s)", flush=True)

    def export_stage(path, si):
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(weights={k: v.cpu() for k, v in M.trainable_state(conn, lm).items()}, stage=si + 1,
                        update=ends[si], n_visual_tokens=store.n_tokens, pool=a.pool), path)
        state["stage_res"][f"stage{si + 1}"]["export_bytes"] = path.stat().st_size

    # one dynamic loss scaler for the whole run (restarting it at 2^16 every stage only adds skipped updates)
    scaler = torch.amp.GradScaler("cuda")
    if ck is not None:
        scaler.load_state_dict(ck["scaler"])
    session_start = state["update"]
    for si, sc in enumerate(stages):
        start, end = (ends[si - 1] if si else 0), ends[si]
        name = f"stage{si + 1}"
        export = rd / "exports" / f"{name}.pt"
        if state["update"] >= end:
            if state["update"] == end and not export.exists():  # crashed between checkpoint and export
                export_stage(export, si)
            continue
        if si >= 1 and not state["lora"]:
            M.add_lora(lm)
            state["lora"] = True
        lm.train(), conn.train()
        opt, sched = make_optimizer(conn, lm, sc, sc["updates"])
        sampler = Sampler(data, si, sc)
        trainable = [p for g in opt.param_groups for p in g["params"]]
        frozen_ok = all((not p.requires_grad) or ("lora_" in n) for n, p in lm.named_parameters())
        sres = state["stage_res"].setdefault(name, dict(
            trainable_connector=sum(p.numel() for p in conn.parameters()),
            trainable_lora=sum(p.numel() for n, p in lm.named_parameters() if p.requires_grad),
            frozen_lm=sum(p.numel() for n, p in lm.named_parameters() if not p.requires_grad),
            only_connector_and_lora_trainable=frozen_ok, train_time_s=0.0, peak_gpu_mem_allocated_bytes=0,
            updates=sc["updates"]))
        assert frozen_ok
        if ck is not None and start < ck["state"]["update"] < end:  # resume inside this stage
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            sampler.load_state(ck["sampler"])
        ck = None
        mix = state["mixture"].setdefault(name, {})
        exc = state["examples"].setdefault(name, {})
        if state["update"] == 0:
            run_dev(0)
        torch.cuda.reset_peak_memory_stats()
        for u in range(state["update"], end):
            t0 = M.now()
            examples, comps = sampler.update(u - start)
            for c in comps:
                mix[c] = mix.get(c, 0) + 1
            for e in examples:
                exc[e["comp"]] = exc.get(e["comp"], 0) + 1
            per_ex, kinds = [], []
            for m in range(C.GRAD_ACCUM):
                mb = examples[m * C.MICRO_BS:(m + 1) * C.MICRO_BS]
                b = M.to_device(make_batch(prompter, store, mb))
                with torch.autocast("cuda", dtype=torch.float16):
                    l = M.example_losses(lm, conn, b)
                scaler.scale(l.sum() / C.EFF_BS).backward()
                per_ex += l.detach().tolist()
                kinds += [e["comp"] for e in mb]
            scaler.unscale_(opt)
            gc = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in conn.named_parameters()}
            if f"grads_{name}" not in state["checks"] and all(v is not None and math.isfinite(v) for v in gc.values()):
                # first update of the stage without FP16 overflow
                lb = [p.grad.norm() for n, p in lm.named_parameters()
                      if p.requires_grad and n.endswith("lora_B") and p.grad is not None]
                state["checks"][f"grads_{name}"] = dict(
                    update=u + 1, connector_grad_norms=gc,
                    connector_grads_nonzero=all(v is not None and v > 0 and math.isfinite(v) for v in gc.values()),
                    lora_B_grad_norm=float(torch.stack(lb).norm()) if lb else None, n_lora_B_with_grad=len(lb))
                print(f"  {name} gradient check: {state['checks'][f'grads_{name}']['connector_grads_nonzero']}")
            gn = torch.nn.utils.clip_grad_norm_(trainable, C.GRAD_CLIP).item()
            lrs = sched.get_last_lr()
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < before
            opt.zero_grad(set_to_none=True)
            sched.step()
            dt = M.now() - t0
            state["update"] = u + 1
            sres["train_time_s"] += dt
            sres["peak_gpu_mem_allocated_bytes"] = max(sres["peak_gpu_mem_allocated_bytes"],
                                                       torch.cuda.max_memory_allocated())
            if skipped:
                state["skipped"].append(u + 1)
            by = defaultdict(list)
            for k, v in zip(kinds, per_ex):
                by[k].append(v)
            append_jsonl(dict(update=u + 1, stage=si + 1, loss=float(np.mean(per_ex)),
                              loss_by_component={k: float(np.mean(v)) for k, v in by.items()},
                              draws=dict(Counter(comps)), lr_connector=lrs[0], lr_lora=lrs[2] if len(lrs) > 2 else None,
                              grad_norm=gn, loss_scale=before, skipped=skipped, time_s=dt), log_path)
            if (u + 1) % 25 == 0 or u == session_start:
                print(f"[{run}] {name} upd {u + 1}/{total} loss {np.mean(per_ex):.4f} lr {lrs[0]:.2e} gn {gn:.2f} "
                      f"{dt:.2f}s/upd mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
                      f"eta {(total - u - 1) * dt / 3600:.2f}h", flush=True)
            if (u + 1) in dev_at:
                run_dev(u + 1)
            if (u + 1) % ckpt_every == 0 and u + 1 != end:
                save_ckpt(opt, sched, scaler, sampler)
            if a.stop_after is not None and u + 1 - session_start >= a.stop_after and u + 1 < total:
                print("stop_after reached; run the same command again to resume")
                return
        export_stage(export, si)
        save_ckpt(opt, sched, scaler, sampler)
        print(f"finished {name} at update {end}; exported {export.name}")

    res = dict(run=run, updates=state["update"], sessions=state["sessions"], calibration=state["calibration"],
               stages=state["stage_res"], mixture_draws=state["mixture"], mixture_examples=state["examples"],
               train_time_s=sum(s["train_time_s"] for s in state["stage_res"].values()),
               dev_eval_time_s=state["dev_time_s"], checkpoint_time_s=state["ckpt_time_s"],
               skipped_updates=state["skipped"], checks=state["checks"], gpu=torch.cuda.get_device_name(0))
    save_json(res, rd / "resources.json")
    print(json.dumps({k: v for k, v in res.items() if k not in ("checks",)}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
