#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO


def image(path: str):
    # Load a model
    model = YOLO("yolo11n-seg.pt")

    image_path = path
    # image = "https://ultralytics.com/images/bus.jpg"

    results = model(image_path)

    print(results[0])

    # results[0].show()

    return 0


def movie(path: str):
    # Load a model
    model = YOLO("yolo11n-seg.pt")

    # Open the video file
    video_path = path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 1

    # Get video properties
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

            # Perform object detection on the frame
            results = model(frame)

            # Get the annotated frame
            annotated_frame = results[0].plot()

            # Write the frame to the output video
            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    return 0
