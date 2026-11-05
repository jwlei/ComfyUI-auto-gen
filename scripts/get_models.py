"""
Downloads the small model the panel runs itself, on the CPU, into models/: MobileSAM, for Select when painting what
to change (pipelines/image/segment.py). Each is checked against its SHA-256, the pins Kokon
uses (github.com/GabrieleLozupone/kokon, crates/qls-core/src/matting.rs). Files already there are left alone.

    python scripts/get_models.py
"""
import hashlib
import os
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = (
    ("https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8/mobile_sam_image_encoder.onnx",
     "models/mobilesam/mobile_sam_image_encoder.onnx", "580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749"),
    ("https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8/sam_mask_decoder_multi.onnx",
     "models/mobilesam/sam_mask_decoder_multi.onnx", "8976b90a87ba50a6a72217a5ff994f7d25ce16f2229fcc1ed259e1294c622ffe"),
)


def fetch(url, path, sha256):
    if os.path.exists(path):
        print(f"✅ {path} is there.")
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    digest = hashlib.sha256()
    part = path + ".part"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != sha256:
        os.remove(part)
        print(f"❌ {path} came down different from what's expected: not kept.")
        return False
    os.replace(part, path)
    print(f"✅ {path}")
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ok = all([fetch(url, os.path.join(ROOT, path), sha256) for url, path, sha256 in FILES])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
