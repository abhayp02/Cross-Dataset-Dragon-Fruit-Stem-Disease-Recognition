"""Flag perceptually similar images across prepared sources for human review."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

from PIL import Image


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        # JPEG decoders can load a small native preview instead of full-resolution pixels.
        image.draft("L", (9, 8))
        pixels = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        values = list(pixels.get_flattened_data())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(values[offset + column] > values[offset + column + 1])
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("data/sources"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/near_duplicate_candidates.csv"))
    parser.add_argument("--max-distance", type=int, default=2)
    args = parser.parse_args()

    records = []
    for manifest_path in sorted(args.sources.glob("*/manifest.csv")):
        root = manifest_path.parent
        for row in csv.DictReader(manifest_path.open(encoding="utf-8")):
            row["image_path"] = str(root / row["file"])
            row["dhash"] = dhash(root / row["file"])
            records.append(row)

    # Any pair at Hamming distance <= 2 must share at least two 16-bit bands.
    buckets: dict[tuple[int, int, int], list[int]] = {}
    candidate_pairs: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        bands = [(record["dhash"] >> shift) & 0xFFFF for shift in (0, 16, 32, 48)]
        for left, right in combinations(range(4), 2):
            key = (left * 4 + right, bands[left], bands[right])
            for other in buckets.get(key, []):
                candidate_pairs.add(tuple(sorted((index, other))))
            buckets.setdefault(key, []).append(index)

    matches = []
    for left, right in sorted(candidate_pairs):
        distance = (records[left]["dhash"] ^ records[right]["dhash"]).bit_count()
        if distance <= args.max_distance:
            first, second = records[left], records[right]
            matches.append({
                "hamming_distance": distance,
                "source_a": first["source_id"],
                "file_a": first["file"],
                "label_a": first["standard_label"],
                "source_b": second["source_id"],
                "file_b": second["file"],
                "label_b": second["standard_label"],
                "same_source": first["source_id"] == second["source_id"],
                "same_label": first["standard_label"] == second["standard_label"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["hamming_distance", "source_a", "file_a", "label_a", "source_b", "file_b", "label_b", "same_source", "same_label"]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matches)
    print(f"images_audited={len(records)}")
    print(f"candidate_pairs={len(matches)}")
    print(f"cross_source_candidates={sum(row['same_source'] is False for row in matches)}")


if __name__ == "__main__":
    main()
