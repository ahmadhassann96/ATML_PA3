"""Task 2 data: COCO captions, annotation-derived grounding pairs, VQAv2 complementary pairs.

    python -m task2.data --select       # choose every example (seed 6304) -> outputs/task2/data/
    python -m task2.data --download     # fetch only the selected COCO images (parallel HTTP)

Inputs (data/task2/raw*, from cocodataset.org and visualqa.org):
  raw/annotations/{captions,instances}_{train,val}2014.json
  raw_vqa/v2_OpenEnded_mscoco_{train,val}2014_questions.json, v2_mscoco_{train,val}2014_annotations.json,
          v2_mscoco_{train,val}2014_complementary_pairs.json

Pair record (one line in the *.jsonl files):
  {pair_id, suite: grounding|vqa, kind: presence|counting|yes/no|number|other, question,
   items: [{qid, split, image_id, target, canonical, refs}, {...}]}
`target` is the training string, `canonical` its normalized form, `refs` the normalized references used
for scoring (ten human answers for VQAv2, the single canonical answer for grounding).
Evaluation images (val2014) are unique across every development and final-test set.
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np

from common.utils import load_json, save_json, write_jsonl
from task2 import config as C
from task2.vqa_norm import consensus, normalize

RAW_VQA = C.DATA_ROOT / "raw_vqa"

# ---- question templates for annotation-derived grounding ----
NOUN = {"tv": "TV", "mouse": "computer mouse", "sports ball": "ball", "remote": "remote control"}
PLURAL = {"person": "people", "bus": "buses", "bench": "benches", "sheep": "sheep", "knife": "knives",
          "sandwich": "sandwiches", "couch": "couches", "wine glass": "wine glasses",
          "toothbrush": "toothbrushes", "mouse": "computer mice", "tv": "TVs", "sports ball": "balls",
          "remote": "remote controls", "scissors": "pairs of scissors", "skis": "pairs of skis",
          "glass": "glasses", "hair drier": "hair driers"}
NO_ARTICLE = {"skis", "scissors"}
# Categories whose instance counts are unreliable as "how many" answers (bunches, florets, stacks, pairs).
COUNT_EXCLUDE = {"skis", "broccoli", "book", "banana", "carrot"}


def presence_question(name):
    n = NOUN.get(name, name)
    art = "" if name in NO_ARTICLE else ("an " if n[0].lower() in "aeiou" else "a ")
    return f"Does this image contain {art}{n}?"


def counting_question(name):
    return f"How many {PLURAL.get(name, name + 's')} can you see?"


# ----------------------------------------------------------------------------- raw loading
def load_coco(split):
    caps = load_json(C.RAW / "annotations" / f"captions_{split}.json")
    inst = load_json(C.RAW / "annotations" / f"instances_{split}.json")
    captions = defaultdict(list)
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        captions[a["image_id"]].append(a["caption"].strip())
    area = {im["id"]: im["width"] * im["height"] for im in inst["images"]}
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    objs = defaultdict(lambda: defaultdict(list))  # image -> category name -> [(area fraction, iscrowd)]
    for a in inst["annotations"]:
        objs[a["image_id"]][cats[a["category_id"]]].append((a["area"] / area[a["image_id"]], a["iscrowd"]))
    return dict(captions=captions, objs=objs, images=sorted(area), cats=[cats[k] for k in sorted(cats)])


def load_vqa(split):
    q = load_json(RAW_VQA / f"v2_OpenEnded_mscoco_{split}_questions.json")["questions"]
    a = load_json(RAW_VQA / f"v2_mscoco_{split}_annotations.json")["annotations"]
    pairs = load_json(RAW_VQA / f"v2_mscoco_{split}_complementary_pairs.json")
    return {x["question_id"]: x for x in q}, {x["question_id"]: x for x in a}, pairs


# ----------------------------------------------------------------------------- VQAv2 pairs
def vqa_candidates(split):
    """Complementary pairs whose two images each have >= 8/10 agreement and different consensus answers."""
    qs, ans, pairs = load_vqa(split)
    out = defaultdict(list)
    stats = Counter()
    for qa, qb in sorted(tuple(sorted(p)) for p in pairs):
        A, B = ans[qa], ans[qb]
        stats["pairs"] += 1
        if qs[qa]["question"] != qs[qb]["question"] or A["image_id"] == B["image_id"]:
            stats["different_question_or_same_image"] += 1
            continue
        items = []
        for x in (A, B):
            raw = [r["answer"] for r in x["answers"]]
            norm = [normalize(r) for r in raw]
            cons, k = consensus(norm)
            # training target: the most frequent original string in the consensus group (ties: first seen)
            group = [r for r, n in zip(raw, norm) if n == cons]
            target = max(group, key=lambda s: (group.count(s), -group.index(s)))
            items.append(dict(qid=x["question_id"], split=split, image_id=x["image_id"], target=target,
                              canonical=cons, refs=norm, agree=k, answer_type=x["answer_type"]))
        if min(it["agree"] for it in items) < C.VQA_MIN_AGREE:
            stats["low_agreement"] += 1
            continue
        if items[0]["canonical"] == items[1]["canonical"]:
            stats["same_consensus"] += 1
            continue
        if items[0]["answer_type"] != items[1]["answer_type"]:
            stats["answer_type_mismatch"] += 1
            continue
        if items[0]["canonical"] == "" or items[1]["canonical"] == "":
            stats["empty_answer"] += 1
            continue
        kind = items[0]["answer_type"]
        for it in items:
            del it["answer_type"]
        out[kind].append(dict(pair_id=f"vqa_{qa}_{qb}", suite="vqa", kind=kind, question=qs[qa]["question"],
                              items=items))
        stats[f"kept_{kind}"] += 1
    return out, dict(stats)


def take_pairs(cands, n, rng, used=None):
    """n random pairs; with `used` (a set of image ids) every image is new and is added to `used`."""
    chosen = []
    for i in rng.permutation(len(cands)):
        p = cands[i]
        ims = [it["image_id"] for it in p["items"]]
        if used is not None and (ims[0] in used or ims[1] in used):
            continue
        chosen.append(p)
        if used is not None:
            used.update(ims)
        if len(chosen) == n:
            break
    assert len(chosen) == n, f"only {len(chosen)} of {n} pairs available"
    return chosen


# ----------------------------------------------------------------------------- grounding pairs
def grounding_index(coco, image_set):
    """Eligible images per category for presence (+/-) and counting (count -> images)."""
    pos, neg, count = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for im in sorted(image_set):
        o = coco["objs"].get(im, {})
        if not o:
            continue  # images without any annotation are never used (more likely unlabeled than empty)
        for c in coco["cats"]:
            inst = o.get(c, [])
            if not inst:
                neg[c].append(im)
                continue
            if any(fr >= C.PRESENCE_MIN_AREA and not crowd for fr, crowd in inst):
                pos[c].append(im)
            if (c not in COUNT_EXCLUDE and C.COUNT_RANGE[0] <= len(inst) <= C.COUNT_RANGE[1]
                    and all(fr >= C.COUNT_MIN_AREA and not crowd for fr, crowd in inst)):
                count[c][len(inst)].append(im)
    return pos, neg, count


def _item(split, im, target, qid):
    t = normalize(target)
    return dict(qid=qid, split=split, image_id=im, target=target, canonical=t, refs=[t])


def presence_pair(split, c, p, n, tag):
    pid = f"presence_{split}_{c.replace(' ', '_')}_{p}_{n}"
    q = presence_question(c)
    return dict(pair_id=pid, suite="grounding", kind="presence", question=q, category=c,
                items=[_item(split, p, "yes", pid + "_a"), _item(split, n, "no", pid + "_b")], tag=tag)


def counting_pair(split, c, a, ka, b, kb, tag):
    pid = f"counting_{split}_{c.replace(' ', '_')}_{a}_{b}"
    q = counting_question(c)
    return dict(pair_id=pid, suite="grounding", kind="counting", question=q, category=c,
                items=[_item(split, a, str(ka), pid + "_a"), _item(split, b, str(kb), pid + "_b")], tag=tag)


def sample_presence(split, idx, c, rng, avoid):
    pos, neg, _ = idx
    P = [i for i in pos[c] if i not in avoid]
    N = [i for i in neg[c] if i not in avoid]
    if not P or not N:
        return None
    return int(P[rng.integers(len(P))]), int(N[rng.integers(len(N))])


def sample_counting(split, idx, c, rng, avoid):
    cnt = {k: [i for i in v if i not in avoid] for k, v in idx[2][c].items()}
    ks = sorted(k for k, v in cnt.items() if v)
    if len(ks) < 2:
        return None
    ka, kb = (int(x) for x in rng.choice(ks, size=2, replace=False))  # two different counts, uniformly
    a = cnt[ka][rng.integers(len(cnt[ka]))]
    b = cnt[kb][rng.integers(len(cnt[kb]))]
    return int(a), ka, int(b), kb


def grounding_eval(split, coco, idx, sizes, rng, used, tag):
    """One pair per category (categories in random order), every image new."""
    out = []
    for kind, n in sizes.items():
        got = 0
        for ci in rng.permutation(len(coco["cats"])):
            c = coco["cats"][ci]
            if kind == "presence":
                s = sample_presence(split, idx, c, rng, used)
                if s:
                    out.append(presence_pair(split, c, *s, tag))
            else:
                s = sample_counting(split, idx, c, rng, used)
                if s:
                    out.append(counting_pair(split, c, *s, tag))
            if s:
                used.update(it["image_id"] for it in out[-1]["items"])
                got += 1
                if got == n:
                    break
        assert got == n, f"{kind}: only {got} of {n} {tag} pairs"
    return out


def grounding_pools(split, coco, idx, rng):
    pools = dict(presence=[], counting=[])
    for c in coco["cats"]:
        for kind in pools:
            seen = set()
            for _ in range(C.POOL_PAIRS_PER_CATEGORY * 5):  # bounded number of attempts
                if len(seen) >= C.POOL_PAIRS_PER_CATEGORY:
                    break
                s = (sample_presence if kind == "presence" else sample_counting)(split, idx, c, rng, ())
                if s is None:
                    break
                key = (s[0], s[-2] if kind == "counting" else s[1])
                if key in seen:
                    continue
                seen.add(key)
                pools[kind].append(presence_pair(split, c, *s, "train") if kind == "presence"
                                   else counting_pair(split, c, *s, "train"))
    return pools


# ----------------------------------------------------------------------------- selection
def select():
    t0 = time.time()
    rng = {k: np.random.default_rng([C.SEED, i]) for i, k in enumerate(
        ["vqa_test", "vqa_dev", "ground_test", "ground_dev", "cap_dev", "cap_train", "vqa_train", "pools"])}
    out = C.data_dir()
    checks = {}

    # ---- evaluation sets from val2014; one shared `used` set keeps all of them image-disjoint
    val = load_coco("val2014")
    vqa_val, checks["vqa_val_filter"] = vqa_candidates("val2014")
    used = set()
    vqa_test = [p for t, n in C.VQA_TEST.items() for p in take_pairs(vqa_val[t], n, rng["vqa_test"], used)]
    vqa_dev = [p for t, n in C.VQA_DEV.items() for p in take_pairs(vqa_val[t], n, rng["vqa_dev"], used)]
    idx_val = grounding_index(val, set(val["images"]))
    g_test = grounding_eval("val2014", val, idx_val, C.GROUND_TEST, rng["ground_test"], used, "test")
    g_dev = grounding_eval("val2014", val, idx_val, C.GROUND_DEV, rng["ground_dev"], used, "dev")
    free = [i for i in val["images"] if i not in used and len(val["captions"][i]) >= 1]
    cap_dev = []
    for i in rng["cap_dev"].choice(len(free), size=C.N_CAP_DEV, replace=False):
        im = free[i]
        caps = val["captions"][im]
        cap_dev.append(dict(image_id=im, split="val2014", caption=caps[rng["cap_dev"].integers(len(caps))]))
        used.add(im)
    del val

    # ---- training sets from train2014
    tr = load_coco("train2014")
    elig = [i for i in tr["images"] if len(tr["captions"][i]) >= C.CAPS_PER_TRAIN_IMAGE]
    cap_train = []
    for i in rng["cap_train"].choice(len(elig), size=C.N_CAP_TRAIN_IMAGES, replace=False):
        im = elig[i]
        caps = tr["captions"][im]
        for j in rng["cap_train"].choice(len(caps), size=C.CAPS_PER_TRAIN_IMAGE, replace=False):
            cap_train.append(dict(image_id=im, split="train2014", caption=caps[j]))
    vqa_tr, checks["vqa_train_filter"] = vqa_candidates("train2014")
    vqa_train = [p for t, n in C.VQA_TRAIN.items() for p in take_pairs(vqa_tr[t], n, rng["vqa_train"])]
    # grounding pools reuse the train2014 images already selected above (no extra downloads)
    pool_images = {c["image_id"] for c in cap_train} | {it["image_id"] for p in vqa_train for it in p["items"]}
    pools = grounding_pools("train2014", tr, grounding_index(tr, pool_images), rng["pools"])
    del tr

    sets = dict(captions_train=cap_train, captions_dev=cap_dev, vqa_train=vqa_train, vqa_dev=vqa_dev,
                vqa_test=vqa_test, ground_pool_presence=pools["presence"], ground_pool_counting=pools["counting"],
                ground_dev=g_dev, ground_test=g_test)
    for k, v in sets.items():
        write_jsonl(v, out / f"{k}.jsonl")

    # ---- images to fetch / encode, and split-isolation checks
    def ims(rows):
        s = set()
        for r in rows:
            for it in r.get("items", [r]):
                s.add((it["split"], it["image_id"]))
        return s

    im_sets = {k: ims(v) for k, v in sets.items()}
    eval_keys = ["captions_dev", "vqa_dev", "vqa_test", "ground_dev", "ground_test"]
    train_keys = [k for k in sets if k not in eval_keys]
    train_ims = set().union(*(im_sets[k] for k in train_keys))
    iso = dict(train_split_only=all(s == "train2014" for s, _ in train_ims),
               eval_split_only=all(s == "val2014" for k in eval_keys for s, _ in im_sets[k]))
    for i, a in enumerate(eval_keys):
        assert not (im_sets[a] & train_ims), a
        for b in eval_keys[i + 1:]:
            assert not (im_sets[a] & im_sets[b]), (a, b)
    for k in eval_keys:  # each evaluation image appears once within its own set too
        n_items = sum(len(r.get("items", [r])) for r in sets[k])
        assert n_items == len(im_sets[k]), k
    iso["eval_sets_pairwise_disjoint_and_disjoint_from_train"] = True
    all_ims = sorted(set().union(*im_sets.values()))
    save_json([dict(split=s, image_id=i) for s, i in all_ims], out / "images.json")

    def kinds(rows):
        return dict(Counter(r["kind"] for r in rows))

    checks.update(
        isolation=iso,
        counts={k: len(v) for k, v in sets.items()},
        kinds={k: kinds(v) for k, v in sets.items() if v and "kind" in v[0]},
        images={k: len(v) for k, v in im_sets.items()},
        n_images_total=len(all_ims), n_train_images=len(train_ims),
        count_answers={k: dict(Counter(it["target"] for p in sets[k] if p["kind"] == "counting" for it in p["items"]))
                       for k in ("ground_pool_counting", "ground_dev", "ground_test")},
        pool_categories={k: len({p["category"] for p in v}) for k, v in pools.items()},
        seconds=time.time() - t0,
    )
    save_json(dict(seed=C.SEED, rule="numpy default_rng([6304, k]) per component; see task2/data.py",
                   ids={k: [r["pair_id"] if "pair_id" in r else r["image_id"] for r in v] for k, v in sets.items()}),
              out / "selection.json")
    save_json(checks, out / "data_checks.json")
    print(json.dumps({k: checks[k] for k in ("counts", "kinds", "images", "n_images_total")}, indent=1))


# ----------------------------------------------------------------------------- image download
def _fetch(split, im):
    p = C.image_path(split, im)
    if p.exists() and p.stat().st_size > 0:
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://images.cocodataset.org/{split}/{p.name}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(p)
            return len(data)
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def download(workers=32, limit=None):
    ims = load_json(C.data_dir() / "images.json")[:limit]
    t0, done, nbytes = time.time(), 0, 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_fetch, r["split"], r["image_id"]) for r in ims]
        for f in cf.as_completed(futs):
            nbytes += f.result()
            done += 1
            if done % 1000 == 0 or done == len(ims):
                print(f"{done}/{len(ims)} images, {nbytes/1e6:.0f} MB new, {time.time()-t0:.0f}s", flush=True)
    save_json(dict(n_images=len(ims), new_bytes=nbytes, seconds=time.time() - t0),
              C.data_dir() / "download_info.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="download only the first N images (tests)")
    a = ap.parse_args()
    if a.select:
        select()
    if a.download:
        download(limit=a.limit)
