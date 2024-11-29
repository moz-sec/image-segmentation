#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ruff: noqa: F401

"""
image-segmentation：performs image recognition with AI
"""

__author__ = "moz-sec"
__version__ = "0.1.0"
__date__ = "2024/11/29 (Created: 2024/11/22)"

import sys

from lib.predict import image, movie, realtime


def main():
    # image("../sample/sample.jpg")
    # movie("../sample/sample.mp4")
    realtime()

    return 0


if __name__ == "__main__":
    sys.exit(main())
