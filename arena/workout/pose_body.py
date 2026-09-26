"""MediaPipe Pose over one clip, streamed. Runs in POSE_PYTHON (mediapipe 0.10.14) like arena/pose_worker.py.

usage: pose_body.py CLIP [STRIDE]

Prints one JSON line per processed frame, as soon as it is processed, so the page can draw the segmentation live:
  {"i": frame, "n": frames, "t": seconds, "lm": [[x, y, visibility], ...13 joints] or null}
Joints (normalised image coordinates): nose, shoulders, elbows, wrists, hips, knees, ankles. No pixels leave this
process.
"""

import json
import sys

import cv2
import mediapipe as mp

KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
clip = sys.argv[1]
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
with mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = None
            if res.pose_landmarks:
                p = res.pose_landmarks.landmark
                lm = [
                    [round(p[k].x, 4), round(p[k].y, 4), round(p[k].visibility, 3)]
                    for k in KEEP
                ]
            print(
                json.dumps({"i": i, "n": n, "t": round(i / fps, 3), "lm": lm}),
                flush=True,
            )
        i += 1
