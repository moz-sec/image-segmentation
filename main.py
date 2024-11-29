#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ruff: noqa: F401

"""
image-segmentation：performs image recognition with AI
"""

__author__ = "moz-sec"
__version__ = "0.1.0"
__date__ = "2024/11/29 (Created: 2024/11/22)"

import argparse
import sys

from instance_segmentation.predict import image, movie, realtime


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns:
        argparse.Namespace: parsed arguments
    """

    parser = argparse.ArgumentParser(description="Perform image segmentation with AI.")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__} ({__date__})",
    )
    parser.add_argument(
        "mode",
        choices=["image", "movie", "realtime"],
        help="Mode of operation: image, movie, or realtime",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the image or movie file (required for image and movie modes)",
    )
    parser.add_argument(
        "--boxes",
        action="store_true",
        default=False,
        help="Draw bounding boxes on detected objects",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.mode == "image":
        if not args.path:
            print("Error: --path is required for image mode")
            return 1
        image(args.path, args.boxes)
    elif args.mode == "movie":
        if not args.path:
            print("Error: --path is required for movie mode")
            return 1
        movie(args.path, args.boxes)
    elif args.mode == "realtime":
        realtime(args.boxes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
