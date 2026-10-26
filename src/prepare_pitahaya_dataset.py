"""Prepare the Converted Images branch of the Dragon Fruit (Pitahaya) dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SOURCE_ID = "dragon_fruit_stem_disease_v1"
PREFIX = "Dragon Fruit (Pitahaya)/Converted Images/"
LABELS = {
    "Anthracnose": "anthracnose",
    "Brown_Stem_Spot": "brown_stem_spot",
    "Gray_Blight": "gray_blight",
    "Healthy": "healthy",
    "Soft_Rot": "soft_rot",
    "Stem_Canker": "stem_canker",
}


def split_records(records: list[dict], seed: int) -> dict[str, list[dict]]:
    records = records.copy()
    random.Random(seed).shuffle(records)
    train_end = round(len(records) * 0.70)
    valid_end = train_end + round(len(records) * 0.15)
    return {"train": records[:train_end], "valid": records[train_end:valid_end], "test": records[valid_end:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path(f"data/sources/{SOURCE_ID}"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.zip_path.is_file():
        raise FileNotFoundError(args.zip_path)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to replace it.")
        shutil.rmtree(args.output)

    rejected: list[dict[str, str]] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(args.zip_path) as archive:
        source_files = sorted(name for name in archive.namelist() if name.startswith(PREFIX) and name.lower().endswith((".jpg", ".jpeg", ".png")))
        for name in source_files:
            original_label = Path(name).parts[-2]
            if original_label not in LABELS:
                rejected.append({"source_path": name, "reason": "unknown_label"})
                continue
            try:
                image_bytes = archive.read(name)
                with Image.open(BytesIO(image_bytes)) as image:
                    image.verify()
                digest = hashlib.sha256(image_bytes).hexdigest()
                groups[digest].append({
                    "source_path": name,
                    "original_label": original_label,
                    "standard_label": LABELS[original_label],
                    "bytes": image_bytes,
                    "sha256": digest,
                })
            except (UnidentifiedImageError, OSError) as error:
                rejected.append({"source_path": name, "reason": f"invalid_image: {error}"})

    accepted_by_label: dict[str, list[dict]] = defaultdict(list)
    for copies in groups.values():
        labels = {copy["standard_label"] for copy in copies}
        if len(labels) > 1:
            for copy in copies:
                rejected.append({"source_path": copy["source_path"], "reason": "conflicting_label_duplicate"})
            continue
        keep = min(copies, key=lambda copy: copy["source_path"])
        accepted_by_label[keep["standard_label"]].append(keep)
        for copy in copies:
            if copy is not keep:
                rejected.append({"source_path": copy["source_path"], "reason": "exact_duplicate"})

    output = args.output.resolve()
    manifest: list[dict[str, str]] = []
    for standard_label, records in sorted(accepted_by_label.items()):
        for split, subset in split_records(records, args.seed).items():
            folder = output / split / standard_label
            folder.mkdir(parents=True, exist_ok=True)
            for record in subset:
                destination = folder / f"{record['sha256'][:16]}.jpg"
                destination.write_bytes(record["bytes"])
                manifest.append({
                    "split": split,
                    "source_id": SOURCE_ID,
                    "source_split": "unsplit_converted_images",
                    "original_label": record["original_label"],
                    "standard_label": standard_label,
                    "file": str(destination.relative_to(output)).replace("\\", "/"),
                    "source_path": record["source_path"],
                    "sha256": record["sha256"],
                })

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "source_id", "source_split", "original_label", "standard_label", "file", "source_path", "sha256"])
        writer.writeheader()
        writer.writerows(manifest)
    with (output / "rejected_files.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_path", "reason"])
        writer.writeheader()
        writer.writerows(rejected)

    counts = defaultdict(lambda: defaultdict(int))
    for row in manifest:
        counts[row["split"]][row["standard_label"]] += 1
    report = {
        "source_zip": str(args.zip_path.resolve()),
        "source_id": SOURCE_ID,
        "included_branch": PREFIX,
        "excluded_branches": ["Original Images", "Annotated Files"],
        "seed": args.seed,
        "usable_images": len(manifest),
        "removed_images": len(rejected),
        "split_counts": {split: dict(labels) for split, labels in counts.items()},
        "removal_reasons": {reason: sum(row["reason"] == reason for row in rejected) for reason in sorted({row["reason"] for row in rejected})},
    }
    (output / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
