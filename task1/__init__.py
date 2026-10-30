import os

# Hugging Face downloads: the Xet transfer client hung indefinitely on our machine, so fall back to plain
# HTTP and put timeouts on every request. Set before huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
