#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""VOC dataset support for the six-class triangle-transfer dataset.

The upstream VOC dataset implementation is tied to the 20 Pascal VOC class
names.  This module keeps the upstream loader and augmentation behavior, but
provides a project-local class mapping and VOC evaluation for arbitrary class
names.
"""

import os
import pickle

import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .voc import AnnotationTransform, VOCDetection


TRIANGLE_CLASSES = (
    "0011c1rod",
    "001jcu5ox",
    "001ob4y7z",
    "001py6kcr",
    "001rq6x2k",
    "001xlffir",
)


class TriangleAnnotationTransform(AnnotationTransform):
    """Map triangle-transfer XML labels to contiguous zero-based IDs."""

    def __init__(self, classes=TRIANGLE_CLASSES, keep_difficult=True):
        class_to_ind = dict(zip(classes, range(len(classes))))
        super().__init__(class_to_ind=class_to_ind, keep_difficult=keep_difficult)


class TriangleVOCDetection(VOCDetection):
    """VOC-style dataset with the six triangle-transfer classes."""

    def __init__(
        self,
        data_dir,
        image_sets=(("2007", "train"),),
        img_size=(416, 416),
        preproc=None,
        classes=TRIANGLE_CLASSES,
        cache=False,
        cache_type="ram",
    ):
        self.triangle_classes = tuple(classes)
        classes_file = os.path.join(data_dir, "classes.txt")
        if os.path.isfile(classes_file):
            with open(classes_file, encoding="utf-8") as class_file:
                file_classes = tuple(
                    line.strip() for line in class_file if line.strip()
                )
            if file_classes != self.triangle_classes:
                raise ValueError(
                    "classes.txt does not match the experiment class order: "
                    f"{file_classes!r} != {self.triangle_classes!r}"
                )
        super().__init__(
            data_dir=data_dir,
            image_sets=list(image_sets),
            img_size=img_size,
            preproc=preproc,
            target_transform=TriangleAnnotationTransform(self.triangle_classes),
            dataset_name="TriangleVOC2007",
            cache=cache,
            cache_type=cache_type,
        )

        # VOCDetection initializes these from Pascal VOC's global 20-class
        # tuple.  Replace the metadata after the parent has loaded annotations.
        self._classes = self.triangle_classes
        self.cats = [
            {"id": idx, "name": name}
            for idx, name in enumerate(self.triangle_classes)
        ]
        self.class_ids = list(range(len(self.triangle_classes)))

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as result_file:
                for image_index, image_id in enumerate(self.ids):
                    image_name = image_id[1]
                    detections = all_boxes[cls_ind][image_index]
                    if detections is None or len(detections) == 0:
                        continue
                    for detection in detections:
                        result_file.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                image_name,
                                detection[-1],
                                detection[0] + 1,
                                detection[1] + 1,
                                detection[2] + 1,
                                detection[3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(self.root, "annotations_cache", "VOC" + self._year, name)
        os.makedirs(cachedir, exist_ok=True)

        aps = []
        use_07_metric = int(self._year) < 2010
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for cls in self._classes:
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps.append(ap)
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as result_file:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, result_file)

        mean_ap = float(np.mean(aps))
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(mean_ap))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(mean_ap))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("--------------------------------------------------------------")

        return mean_ap
