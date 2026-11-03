"""Cache frozen SigLIP patch features for every selected image (FP16), then free the encoder.

    python -m task2.features            # -> outputs/task2/cache/features.npy + features_index.json
    python -m task2.features --fake     # random features of the right shape (local smoke tests only)

Features are SigLIP's `last_hidden_state`: the 196 final patch tokens in 14 x 14 row-major order, after the
vision tower's post-layernorm, which SiglipVisionTransformer applies exactly once (we never apply it
again). Preprocessing is SiglipImageProcessor's fixed resize to 224 x 224 (bilinear), rescale and
normalization -- no random crops or flips -- on RGB-converted images. The encoder runs in FP32; the cache is FP16.
"""
import argparse
import concurrent.futures as cf
import time

import numpy as np
import torch
from PIL import Image

from common.utils import env_info, load_json, save_json
from task2 import config as C


def key(split, image_id):
    return f"{split}/{image_id}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    ims = load_json(C.data_dir() / "images.json")
    C.cache_dir().mkdir(parents=True, exist_ok=True)
    path = C.cache_dir() / "features.npy"
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(ims), C.N_VIS, C.D_VIS))
    index = {key(r["split"], r["image_id"]): i for i, r in enumerate(ims)}
    t0 = time.perf_counter()
    if a.fake:
        rng = np.random.default_rng(C.SEED)
        for s in range(0, len(ims), 1024):
            arr[s:s + 1024] = rng.standard_normal((min(1024, len(ims) - s), C.N_VIS, C.D_VIS), dtype=np.float32)
        info = dict(fake=True)
    else:
        from transformers import SiglipImageProcessor, SiglipVisionModel
        proc = SiglipImageProcessor.from_pretrained(C.VISION_ID)
        torch.cuda.reset_peak_memory_stats()
        t_load = time.perf_counter()
        enc = SiglipVisionModel.from_pretrained(C.VISION_ID, torch_dtype=torch.float32).cuda().eval()
        load_s = time.perf_counter() - t_load

        def prep(rows):
            pics = [Image.open(C.image_path(r["split"], r["image_id"])).convert("RGB") for r in rows]
            return proc(images=pics, return_tensors="pt")["pixel_values"]

        batches = [ims[s:s + a.batch] for s in range(0, len(ims), a.batch)]
        with cf.ThreadPoolExecutor(4) as ex, torch.no_grad():
            futs = [ex.submit(prep, b) for b in batches[:3]]
            for bi, b in enumerate(batches):
                pv = futs[bi].result()
                if bi + 3 < len(batches):
                    futs.append(ex.submit(prep, batches[bi + 3]))
                out = enc(pixel_values=pv.cuda()).last_hidden_state
                assert out.shape[1:] == (C.N_VIS, C.D_VIS), out.shape
                arr[bi * a.batch: bi * a.batch + len(b)] = out.half().cpu().numpy()
                if bi % 20 == 0:
                    print(f"{bi * a.batch + len(b)}/{len(ims)} images, {time.perf_counter() - t0:.0f}s", flush=True)
        info = dict(fake=False, model=C.VISION_ID, encoder_load_s=load_s, batch=a.batch,
                    peak_gpu_mem_allocated_bytes=torch.cuda.max_memory_allocated(),
                    preprocessing=proc.to_dict(), env=env_info())
        del enc
        torch.cuda.empty_cache()
    arr.flush()
    sample = np.asarray(arr[: min(256, len(ims))], dtype=np.float32)
    info.update(n_images=len(ims), shape=[len(ims), C.N_VIS, C.D_VIS], dtype="float16",
                seconds=time.perf_counter() - t0, bytes=path.stat().st_size,
                finite=bool(np.isfinite(sample).all()), mean_token_norm_sample=float(np.linalg.norm(sample, axis=-1).mean()))
    save_json(index, C.cache_dir() / "features_index.json")
    save_json(info, C.data_dir() / ("feature_cache_info_fake.json" if a.fake else "feature_cache_info.json"))
    print({k: v for k, v in info.items() if k not in ("preprocessing", "env")})


class FeatureStore:
    """Memory-mapped FP16 features; pool=2 averages each 2 x 2 block of the 14 x 14 grid (49 tokens)."""

    def __init__(self, pool=1):
        self.arr = np.load(C.cache_dir() / "features.npy", mmap_mode="r")
        self.index = load_json(C.cache_dir() / "features_index.json")
        self.pool = pool
        self.n_tokens = C.N_VIS // (pool * pool)

    def get(self, split, image_id):
        f = torch.from_numpy(np.array(self.arr[self.index[key(split, image_id)]]))  # copy out of the read-only map
        if self.pool > 1:
            g, p = 14, self.pool
            f = f.float().reshape(g // p, p, g // p, p, C.D_VIS).mean((1, 3)).reshape(-1, C.D_VIS).half()
        return f


if __name__ == "__main__":
    main()
