"""SigLIP-Qwen VLM pieces: connector, prompt layout, batch assembly, loss, greedy decoding.

Layout of one example (Qwen3 chat template, enable_thinking=False):
  <|im_start|>user\\n [Z_1 ... Z_n] \\n <question tokens> <|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
  <target tokens> <|im_end|>
The n visual embeddings Z = s * g_phi(F(I)) replace n placeholder positions inside the user message,
before the question. Labels cover the target tokens and the end-of-turn token only. For the no-image
condition the visual tokens and their separating newline are removed and masks/positions rebuilt, so
the prompt is exactly the text-only chat prompt.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from task1 import adapters
from task2 import config as C

MARK, QMARK = "<<IMAGE>>", "<<QUESTION>>"


class Connector(nn.Module):
    """Linear(768 -> 1024), GELU, Linear(1024 -> 1024), times a fixed calibration scale s."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(C.D_VIS, C.D_LM)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(C.D_LM, C.D_LM)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, f):
        return self.fc2(self.act(self.fc1(f))) * self.scale


class Prompter:
    def __init__(self, tok, n_vis):
        self.tok, self.n_vis = tok, n_vis
        text = tok.apply_chat_template([{"role": "user", "content": MARK + QMARK}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        pre, rest = text.split(MARK)
        mid, post = rest.split(QMARK)
        assert mid == "", text
        enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
        self.template_text = text
        self.pre, self.post, self.sep = enc(pre), enc(post), enc("\n")
        self.suffix = enc("\n" + C.SHORT_ANSWER_SUFFIX)
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.placeholder = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.enc = enc

    def encode(self, question, image, target=None, kind="answer", add_suffix=True):
        """question: raw text (capped at 80 content tokens); kind: 'caption' | 'answer' selects target cap."""
        q = self.enc(question)[: C.CAPS["question"]] + (self.suffix if add_suffix else [])
        vis = [self.placeholder] * self.n_vis + self.sep if image else []
        ids = self.pre + vis + q + self.post
        vis_mask = [False] * len(self.pre) + [True] * (self.n_vis if image else 0) + \
                   [False] * (len(ids) - len(self.pre) - (self.n_vis if image else 0))
        labels = [-100] * len(ids)
        n_prompt = len(ids)
        if target is not None:
            t = self.enc(target)[: C.CAPS["caption" if kind == "caption" else "answer"]] + [self.im_end]
            ids, labels, vis_mask = ids + t, labels + t, vis_mask + [False] * len(t)
        return dict(ids=ids, labels=labels, vis_mask=vis_mask, n_prompt=n_prompt)


def collate(encs, feats, pad_id, left=False, drop_termination=False):
    """encs: Prompter.encode outputs; feats: list of [n_vis, 768] tensors (None for no-image examples)."""
    T = max(len(e["ids"]) for e in encs)
    B = len(encs)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    vm = torch.zeros((B, T), dtype=torch.bool)
    for i, e in enumerate(encs):
        n = len(e["ids"])
        sl = slice(T - n, T) if left else slice(0, n)
        ids[i, sl] = torch.tensor(e["ids"])
        l = torch.tensor(e["labels"])
        if drop_termination:  # answer NLL excludes the end-of-turn token
            l[(l != -100).nonzero()[-1]] = -100
        lab[i, sl] = l
        att[i, sl] = 1
        vm[i, sl] = torch.tensor(e["vis_mask"])
    pos = (att.cumsum(1) - 1).clamp(min=0)
    f = [x for x in feats if x is not None]
    f = torch.stack(f) if f else None
    return dict(ids=ids, labels=lab, att=att, pos=pos, vis_mask=vm, feats=f)


def embed(lm, connector, batch):
    """Text embeddings with visual tokens scattered into the placeholder positions (keeps the graph to phi)."""
    e = lm.get_input_embeddings()(batch["ids"])
    if batch["feats"] is not None:
        z = connector(batch["feats"].float()).to(e.dtype)
        e = e.masked_scatter(batch["vis_mask"].unsqueeze(-1), z.reshape(-1, z.shape[-1]))
    return e


def to_device(batch, dev="cuda"):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def example_losses(lm, connector, batch):
    """Per-example mean token cross-entropy over supervised positions -> tensor [B] (FP32)."""
    e = embed(lm, connector, batch)
    h = lm.model(inputs_embeds=e, attention_mask=batch["att"], position_ids=batch["pos"],
                 use_cache=False).last_hidden_state
    h, tgt = h[:, :-1], batch["labels"][:, 1:]
    m = tgt != -100
    ce = F.cross_entropy(lm.lm_head(h[m]).float(), tgt[m], reduction="none")
    rows = m.nonzero()[:, 0]
    B = tgt.shape[0]
    s = torch.zeros(B, device=ce.device, dtype=torch.float32).index_add_(0, rows, ce)
    return s / m.sum(1).clamp(min=1)


@torch.no_grad()
def greedy(lm, connector, batch, max_new, stop_ids):
    """Greedy decoding with a KV cache for a left-padded batch; returns token lists (stop token removed)."""
    e = embed(lm, connector, batch)
    att, pos = batch["att"], batch["pos"]
    out = lm(inputs_embeds=e, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1)
    B = nxt.shape[0]
    stop = torch.tensor(sorted(stop_ids), device=nxt.device)
    done = torch.zeros(B, dtype=torch.bool, device=nxt.device)
    toks = [[] for _ in range(B)]
    last = pos[:, -1]
    for step in range(max_new):
        is_stop = torch.isin(nxt, stop)
        for i in range(B):
            if not done[i] and not is_stop[i]:
                toks[i].append(int(nxt[i]))
        done |= is_stop
        if done.all() or step == max_new - 1:
            break
        att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=att.device)], 1)
        last = last + 1
        out = lm(input_ids=nxt[:, None], attention_mask=att, position_ids=last[:, None], past_key_values=past,
                 use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
    return toks


# ----------------------------------------------------------------------------- loading
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(C.LM_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_lm():
    lm = AutoModelForCausalLM.from_pretrained(C.LM_ID, torch_dtype=torch.float16, attn_implementation="sdpa").cuda()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm


def add_lora(lm):
    return adapters.inject(lm, C.LORA["targets"], C.LORA["r"], C.LORA["alpha"], C.LORA["dropout"], C.SEED)


def trainable_state(connector, lm):
    st = {"connector." + k: v.detach().clone() for k, v in connector.state_dict().items()}
    for n, p in lm.named_parameters():
        if p.requires_grad:
            st["lm." + n] = p.detach().clone()
    return st


def load_trainable(connector, lm, st):
    connector.load_state_dict({k[len("connector."):]: v for k, v in st.items() if k.startswith("connector.")})
    named = dict(lm.named_parameters())
    for k, v in st.items():
        if k.startswith("lm."):
            named[k[3:]].data.copy_(v)


def load_final(run_dir, stage="stage3"):
    """Eval model from an exported stage: frozen FP16 Qwen + LoRA (stage >= 2) + connector, eval mode."""
    st = torch.load(run_dir / "exports" / f"{stage}.pt", map_location="cpu")
    tok, lm = load_tokenizer(), load_lm()
    if any(k.startswith("lm.") for k in st["weights"]):
        add_lora(lm)
    connector = Connector().cuda()
    load_trainable(connector, lm, {k: v.cuda() for k, v in st["weights"].items()})
    lm.eval(), connector.eval()
    return tok, lm, connector, st


def now():
    torch.cuda.synchronize()
    return time.perf_counter()
