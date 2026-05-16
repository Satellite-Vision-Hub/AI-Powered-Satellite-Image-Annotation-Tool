"""
Whitelist-based zipper for Colab upload.
Only includes the project SOURCE CODE — excludes venv, data, git, caches.
"""
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP   = PROJECT_ROOT / "skylogic_code.zip"

# Whitelist — these top-level items get included
TOP_LEVEL_DIRS = [
    "scripts",
    "docker",
    "docs",
]
TOP_LEVEL_FILES = [
    "requirements.txt",
    ".env.example",
    "README.md",
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    "yolov8n.pt",
]

# Inside skylogic/, only these subpackages (NOT the venv internals)
SKYLOGIC_SUBPACKAGES = [
    "agents",
    "api",
    "ensemble",
    "ingestion",
    "models",
    "vector_store",
]
SKYLOGIC_FILES = [
    "__init__.py",
    "config.py",
    "database.py",
]

# Always skip these (defence in depth)
SKIP_PATTERNS = {
    "__pycache__", ".pyc", ".pyo", ".pyd",
    "Lib", "Scripts", "Include", "share", "Tcl",
    ".git", ".vscode", ".pytest_cache",
    "pyvenv.cfg", ".env", "skylogic.db",
}


def should_skip(path: Path) -> bool:
    """True if any part of the path matches a skip pattern."""
    for part in path.parts:
        if part in SKIP_PATTERNS:
            return True
        if part.endswith((".pyc", ".pyo", ".pyd", ".db")):
            return True
    return False


def add_dir(zf: zipfile.ZipFile, src: Path, arc_base: str) -> int:
    """Add a directory recursively. Returns file count."""
    count = 0
    if not src.exists():
        print(f"  [skip] {arc_base}/ — does not exist")
        return 0
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]
        for fn in files:
            fp = root_p / fn
            if should_skip(fp):
                continue
            rel = fp.relative_to(src)
            arc = f"{arc_base}/{rel.as_posix()}"
            zf.write(fp, arc)
            count += 1
    return count


def main():
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
        print(f"Removed old zip.")

    print(f"Building {OUTPUT_ZIP} ...")
    total = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:

        # Top-level files
        for fname in TOP_LEVEL_FILES:
            fp = PROJECT_ROOT / fname
            if fp.exists():
                zf.write(fp, fname)
                total += 1
                print(f"  [file] {fname}")

        # Top-level directories
        for dname in TOP_LEVEL_DIRS:
            dp = PROJECT_ROOT / dname
            if dp.exists():
                n = add_dir(zf, dp, dname)
                total += n
                print(f"  [dir ] {dname}/ — {n} files")

        # skylogic package (whitelist subdirs/files only)
        sky = PROJECT_ROOT / "skylogic"
        if sky.exists():
            for fname in SKYLOGIC_FILES:
                fp = sky / fname
                if fp.exists():
                    zf.write(fp, f"skylogic/{fname}")
                    total += 1
                    print(f"  [file] skylogic/{fname}")
            for sub in SKYLOGIC_SUBPACKAGES:
                sp = sky / sub
                if sp.exists():
                    n = add_dir(zf, sp, f"skylogic/{sub}")
                    total += n
                    print(f"  [dir ] skylogic/{sub}/ — {n} files")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print()
    print(f"Total files : {total}")
    print(f"Zip size    : {size_mb:.2f} MB")
    print(f"Zip path    : {OUTPUT_ZIP}")

    if size_mb > 100:
        print("\nWARNING: Zip is unexpectedly large — venv internals may have leaked.")
        sys.exit(1)


if __name__ == "__main__":
    main()
