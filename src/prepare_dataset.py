"""Create a clean, deduplicated dragon-fruit stem dataset from a Roboflow ZIP."""

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


SPLITS = (("train", 0.70), ("valid", 0.15), ("test", 0.15))
STANDARD_LABELS = {
    "anthracnose": "anthracnose",
    "cactusvirusx": "cactus_virus_x",
    "healthy": "healthy",
    "stemcanker": "stem_canker",
}


def image_files(archive: zipfile.ZipFile) -> list[str]:
    return sorted(name for name in archive.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png")))


def class_from_path(name: str) -> str:
    parts = Path(name).parts
    if len(parts) < 3 or parts[0] not in {"train", "valid", "test"}:
        raise ValueError(f"Unexpected image path: {name}")
    return parts[1].strip().lower().replace(" ", "_")


def split_items(items: list[dict], seed: int) -> dict[str, list[dict]]:
    shuffled = items.copy()
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = round(total * SPLITS[0][1])
    valid_end = train_end + round(total * SPLITS[1][1])
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/dataset"))
    parser.add_argument("--source-id", default="common_dragon_fruit_stem_disease_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.zip_path.is_file():
        raise FileNotFoundError(args.zip_path)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to replace it.")
        shutil.rmtree(args.output)

    output = args.output.resolve()
    rejected: list[dict[str, str]] = []
    groups: dict[str, list[dict]] = defaultdict(list)

    with zipfile.ZipFile(args.zip_path) as archive:
        for name in image_files(archive):
            try:
                image_bytes = archive.read(name)
                with Image.open(BytesIO(image_bytes)) as image:
                    image.verify()
                groups[hashlib.sha256(image_bytes).hexdigest()].append(
                    {"source_path": name, "class": class_from_path(name), "bytes": image_bytes}
                )
            except (UnidentifiedImageError, OSError, ValueError) as error:
                rejected.append({"source_path": name, "reason": f"invalid_image: {error}"})

        accepted: list[dict] = []
        for digest, copies in groups.items():
            labels = {copy["class"] for copy in copies}
            if len(labels) > 1:
                for copy in copies:
                    rejected.append({"source_path": copy["source_path"], "reason": "conflicting_label_duplicate"})
                continue
            keep = min(copies, key=lambda copy: copy["source_path"])
            accepted.append({**keep, "sha256": digest})
            for copy in copies:
                if copy is not keep:
                    rejected.append({"source_path": copy["source_path"], "reason": "exact_duplicate"})

    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in accepted:
        by_class[item["class"]].append(item)

    manifest: list[dict[str, str]] = []
    for class_name, items in sorted(by_class.items()):
        for split_name, split_items_for_class in split_items(items, args.seed).items():
            folder = output / split_name / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for item in split_items_for_class:
                filename = f"{item['sha256'][:16]}.jpg"
                destination = folder / filename
                destination.write_bytes(item["bytes"])
                manifest.append(
                    {
                        "split": split_name,
                        "source_id": args.source_id,
                        "source_split": item["source_path"].split("/", maxsplit=1)[0],
                        "original_label": class_name,
                        "standard_label": STANDARD_LABELS.get(class_name, class_name),
                        "file": str(destination.relative_to(output)).replace("\\", "/"),
                        "source_path": item["source_path"],
                        "sha256": item["sha256"],
                    }
                )

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "source_id", "source_split", "original_label", "standard_label", "file", "source_path", "sha256"],
        )
        writer.writeheader()
        writer.writerows(manifest)
    with (output / "rejected_files.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_path", "reason"])
        writer.writeheader()
        writer.writerows(rejected)

    counts = {split: {} for split, _ in SPLITS}
    for row in manifest:
        counts[row["split"]][row["standard_label"]] = counts[row["split"]].get(row["standard_label"], 0) + 1
    report = {
        "source_zip": str(args.zip_path.resolve()),
        "source_id": args.source_id,
        "seed": args.seed,
        "classes": sorted(by_class),
        "usable_images": len(manifest),
        "removed_images": len(rejected),
        "split_counts": counts,
        "removal_reasons": {reason: sum(row["reason"] == reason for row in rejected) for reason in sorted({row["reason"] for row in rejected})},
    }
    (output / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
