#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
image-segmentation：performs image recognition with AI
"""

__author__ = "moz-sec"
__version__ = "0.1.0"
__date__ = "2024/11/23 (Created: 2024/11/22)"

import sys

from ultralytics import YOLO


def main():
    # Load a model
    # model = YOLO("yolo11n.pt")
    model = YOLO("yolo11n-seg.pt")

    # training(model)

    results = model("../sample/example.jpg")
    # results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()

    # Export the model to ONNX format
    # model.export(format="onnx")  # return path to exported model

    return 0


if __name__ == "__main__":
    sys.exit(main())
