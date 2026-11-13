"""Download CUAD v1 (SQuAD-format official split + helper files) into config.DATA_RAW.

Safe to re-run: files that already exist are skipped.
Usage:  python -m src.download_data
"""
import urllib.request
import zipfile
from pathlib import Path

from src import config

REPO_RAW = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main"
FILES = {
    "data.zip": "https://github.com/TheAtticusProject/cuad/raw/main/data.zip",
    "evaluate.py": f"{REPO_RAW}/evaluate.py",
    "category_descriptions.csv": f"{REPO_RAW}/category_descriptions.csv",
}


def download(url: str, dest: Path) -> None:
    """Download url to dest unless dest already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading:   {dest.name}")
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)  # only a complete download gets the final name


def main() -> None:
    config.ensure_dirs()
    raw = config.DATA_RAW
    print(f"Saving to: {raw}")
    for name, url in FILES.items():
        download(url, raw / name)

    zip_path = raw / "data.zip"
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{zip_path} is not a valid zip file. Delete it and re-run.")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not (raw / member).exists():
                zf.extract(member, raw)

    print("\nFiles in data/raw:")
    for p in sorted(raw.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(raw)}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()