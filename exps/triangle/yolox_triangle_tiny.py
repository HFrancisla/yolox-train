#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""YOLOX-tiny experiment for the six-class triangle-transfer dataset."""

import os

from yolox.data import (
    TrainTransform,
    TriangleVOCDetection,
    ValTransform,
    get_yolox_datadir,
)
from yolox.evaluators import VOCEvaluator
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # YOLOX-tiny model shape and the historical training image size.
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.test_size = (416, 416)

        # The old run used the VOC-style six-class task and disabled MixUp in
        # the tiny default experiment.  Keep the rest of YOLOX's defaults.
        self.enable_mixup = False
        self.warmup_epochs = 1
        self.data_num_workers = min(4, os.cpu_count() or 1)
        self.test_conf = 0.1
        self.nmsthre = 0.5
        self.exp_name = "yolox_triangle_tiny"

        repo_data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "triangle-2851")
        )
        default_data_dir = repo_data_dir
        if not os.path.isdir(default_data_dir):
            default_data_dir = os.path.join(get_yolox_datadir(), "triangle-2851")
        self.data_dir = os.path.abspath(
            os.environ.get("TRIANGLE_DATA_DIR", default_data_dir)
        )

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        return TriangleVOCDetection(
            data_dir=self.data_dir,
            image_sets=[("2007", "train")],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        legacy = kwargs.get("legacy", False)
        return TriangleVOCDetection(
            data_dir=self.data_dir,
            image_sets=[("2007", "val")],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        return VOCEvaluator(
            dataloader=self.get_eval_loader(
                batch_size,
                is_distributed,
                testdev=testdev,
                legacy=legacy,
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
