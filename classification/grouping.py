"""Group images that would leak across a split.

BUSI filenames are not patient identifiers. Exact byte matches and close
perceptual hashes are grouped instead. A group whose files carry more than
one folder label is a label conflict and is held out of every split.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).astype(np.uint8).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def group_records(records: list[dict], hamming_max: int = 5) -> dict:
    """``records`` need image_id, label, and image_path.

    Returns groups, conflict groups, and the pair list used to connect them.
    """
    hashes: dict[str, list[str]] = defaultdict(list)
    codes: dict[str, int] = {}
    labels: dict[str, str] = {}
    for record in records:
        image_id = record["image_id"]
        labels[image_id] = record["label"]
        hashes[file_sha256(Path(record["image_path"]))].append(image_id)
        codes[image_id] = dhash(Path(record["image_path"]))

    components = UnionFind()
    for image_id in labels:
        components.find(image_id)
    pairs = []
    for duplicates in hashes.values():
        for other in duplicates[1:]:
            components.union(duplicates[0], other)
            pairs.append({"image_a": duplicates[0], "image_b": other, "dhash_hamming": 0, "exact": True})
    ids = list(labels)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            distance = (codes[left] ^ codes[right]).bit_count()
            if 0 < distance <= hamming_max:
                components.union(left, right)
                pairs.append(
                    {
                        "image_a": left,
                        "image_b": right,
                        "dhash_hamming": distance,
                        "exact": False,
                        "cross_label": labels[left] != labels[right],
                    }
                )

    grouped: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        grouped[components.find(image_id)].append(image_id)
    groups = []
    conflicts = []
    for members in grouped.values():
        member_labels = sorted({labels[member] for member in members})
        payload = {"members": sorted(members), "labels": member_labels}
        if len(member_labels) > 1:
            conflicts.append(payload)
        else:
            groups.append({**payload, "label": member_labels[0]})
    groups.sort(key=lambda group: (group["label"], group["members"][0]))
    conflicts.sort(key=lambda group: group["members"][0])
    return {"groups": groups, "conflicts": conflicts, "pairs": pairs, "labels": labels}
