"""
Take the raw dataset —

    datasets/
      <site_1>/ *.jpg
      <site_2>/ *.jpg
      ...

and produce a stratified train/val/test split —  split into 70/15/15 

    data/
      train/<site_1>/...   val/<site_1>/...   test/<site_1>/...
      train/<site_2>/...   val/<site_2>/...   test/<site_2>/...

Key design choices
------------------
* STRATIFIED: each class is split on its own, so every site appears in all
  three sets even when it has only ~29 photos. A naive global shuffle could
  leave a rare site out of val/test entirely — fatal for evaluation.
* COPIES, never moves: your raw datasets/ folder stays untouched as the
  source of truth. Re-run any time; use --clean to rebuild from scratch.
* SEEDED: same seed -> same split every run, so results are comparable.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def list_images(folder: Path) -> list[Path]:
    """All image files directly inside `folder`, sorted for determinism."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def split_counts(n: int, train: float, val: float) -> tuple[int, int, int]:
    """
    Turn a class size `n` into (n_train, n_val, n_test).

    Guards for tiny classes: if the class has at least 3 images, we force at
    least 1 into val and 1 into test, so no class is missing from evaluation
    just because rounding sent everything to train.
    """
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val

    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
    return n_train, n_val, n_test


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified train/val/test split.")
    ap.add_argument("--src", default="datasets", help="raw data root")
    ap.add_argument("--dst", default="data", help="output root")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true",
                    help="delete existing train/val/test before splitting")
    args = ap.parse_args()

    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0 (got {ratio_sum}).")

    random.seed(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src.resolve()}")

    if args.clean:
        for split in ("train", "val", "test"):
            shutil.rmtree(dst / split, ignore_errors=True)

    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class subfolders found inside {src.resolve()}")

    print(f"{'class':<28} {'total':>6} {'train':>6} {'val':>5} {'test':>5}")
    print("-" * 54)

    totals = {"train": 0, "val": 0, "test": 0}
    warnings: list[str] = []

    for cls in class_dirs:
        images = list_images(cls)
        n = len(images)
        if n == 0:
            warnings.append(f"  {cls.name}: no images — skipped")
            continue

        random.shuffle(images)
        n_train, n_val, n_test = split_counts(n, args.train, args.val)

        buckets = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, files in buckets.items():
            out_dir = dst / split / cls.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out_dir / f.name)
            totals[split] += len(files)

        print(f"{cls.name:<28} {n:>6} {n_train:>6} {n_val:>5} {n_test:>5}")

        if n < 20:
            warnings.append(f"  {cls.name}: only {n} images — val/test will be very small")

    print("-" * 54)
    grand = sum(totals.values())
    print(f"{'TOTAL':<28} {grand:>6} {totals['train']:>6} "
          f"{totals['val']:>5} {totals['test']:>5}")
    print(f"\nclasses: {len(class_dirs)}   images copied: {grand}")
    print(f"output: {dst.resolve()}")

    if warnings:
        print("\nnotes:")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()