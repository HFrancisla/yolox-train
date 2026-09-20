#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Prepare the triangle-transfer images/XML files as a YOLOX VOC dataset."""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def copy_split(data_dir: Path, train_ratio: float, seed: int):
    source_images = data_dir / "JPEGImages"
    source_annotations = data_dir / "Annotations"
    classes_file = data_dir / "classes.txt"

    if not source_images.is_dir():
        raise FileNotFoundError(f"missing image directory: {source_images}")
    if not source_annotations.is_dir():
        raise FileNotFoundError(f"missing annotation directory: {source_annotations}")
    if not classes_file.is_file():
        raise FileNotFoundError(f"missing classes file: {classes_file}")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    image_by_stem = {
        path.stem: path
        for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    annotation_by_stem = {
        path.stem: path
        for path in source_annotations.glob("*.xml")
        if path.is_file()
    }
    stems = sorted(set(image_by_stem) & set(annotation_by_stem))
    if not stems:
        raise RuntimeError("no image/XML pairs found")

    rng = random.Random(seed)
    rng.shuffle(stems)
    train_count = int(len(stems) * train_ratio)
    train_stems = sorted(stems[:train_count])
    val_stems = sorted(stems[train_count:])

    voc_root = data_dir / "VOC2007"
    voc_images = voc_root / "JPEGImages"
    voc_annotations = voc_root / "Annotations"
    image_sets = voc_root / "ImageSets" / "Main"
    for directory in (voc_images, voc_annotations, image_sets):
        directory.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image_path = image_by_stem[stem]
        annotation_path = annotation_by_stem[stem]
        target_image = voc_images / f"{stem}.jpg"
        target_annotation = voc_annotations / f"{stem}.xml"
        if image_path.suffix.lower() == ".jpg":
            shutil.copy2(image_path, target_image)
        else:
            raise ValueError(
                f"YOLOX's VOC loader expects .jpg files; found {image_path.name}"
            )
        shutil.copy2(annotation_path, target_annotation)

    def write_ids(filename, ids):
        (image_sets / filename).write_text(
            "".join(f"{stem}\n" for stem in ids), encoding="utf-8"
        )

    write_ids("train.txt", train_stems)
    write_ids("val.txt", val_stems)
    write_ids("trainval.txt", sorted(stems))

    print(f"dataset: {data_dir}")
    print(f"images with annotations: {len(stems)}")
    print(f"train: {len(train_stems)}")
    print(f"val: {len(val_stems)}")
    print(f"seed: {seed}")
    print(f"VOC root: {voc_root}")


if __name__ == "__main__":
    args = parse_args()
    copy_split(args.data_dir, args.train_ratio, args.seed)
