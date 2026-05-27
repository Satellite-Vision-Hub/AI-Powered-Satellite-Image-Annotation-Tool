#!/usr/bin/env python3
"""
zip_sample_dataset.py - Package sample_dataset_9k/ into a single zip for
fast, reliable upload to Google Drive.

Uploading one ~3 GB zip is far more reliable than uploading 9,000+ loose
files (Drive folder uploads frequently stall). The Colab notebook detects
and unzips this archive automatically.

ZIP_STORED is used (no compression) because PNG patches are already
compressed - storing is fast and the size penalty is ~0.

Usage:
    python scripts/zip_sample_dataset.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
SRC = PROJ / "sample_dataset_9k"
OUT = PROJ / "sample_dataset_9k.zip"


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found - run build_sample_dataset.py first")
        return 2

    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    print(f"Zipping {len(files)} files -> {OUT.name}")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, f in enumerate(files, 1):
            # arcname keeps the sample_dataset_9k/ prefix so unzip recreates it
            zf.write(f, f.relative_to(PROJ))
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}")

    mb = OUT.stat().st_size / 1024 / 1024
    print(f"Done: {OUT}  ({mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
