#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO


def image(path: str) -> int:
    """image segmentation

    Args:
        path (str): path to the image file

    Returns:
        int: 0 if successful, 1 if error occurs
    """
    model = YOLO("yolo11n-seg.pt")

    results = model(path)
    results[0].show()

    return 0


def movie(path: str) -> int:
    """movie segmentation

    Args:
        path (str): path to the movie file

    Returns:
        int: 0 if successful, 1 if error occurs
    """

    model = YOLO("yolo11n-seg.pt")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video {path}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            annotated_frame = results[0].plot()

            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def realtime() -> int:
    """real-time image segmentation

    Returns:
        int: 0 if successful, 1 if error occurs
    """
    model = YOLO("yolo11n-seg.pt")

    # Open the camera stream (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.imshow("Real-time Instance Segmentation(Quit by 'q')", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0
