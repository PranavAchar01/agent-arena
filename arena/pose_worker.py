"""MediaPipe hand tracking for one clip window. Runs in its own interpreter (POSE_PYTHON, mediapipe 0.10.14 on
Python 3.11) because newer mediapipe releases abort on macOS. Writes only numbers: per frame, for each hand, the
wrist, thumb tip and index tip in normalised image coordinates. No pixels are kept.

usage: pose_worker.py CLIP START END OUT.json MODEL
"""

import json
import sys

import cv2
import mediapipe as mp

clip, start, end, out, model = (
    sys.argv[1],
    float(sys.argv[2]),
    float(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
)
opts = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frames = []
with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t < start:
            continue
        if t > end:
            break
        img = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        )
        r = lm.detect_for_video(img, int(t * 1000))
        hands = []
        for k, pts in enumerate(r.hand_landmarks):
            side = r.handedness[k][0].category_name if r.handedness else "?"
            pick = lambda j: [round(pts[j].x, 4), round(pts[j].y, 4)]  # noqa: E731
            hands.append(
                {
                    "side": side,
                    "wrist": pick(0),
                    "thumb": pick(4),
                    "index": pick(8),
                    "mcp": pick(9),
                }
            )
        frames.append({"t": round(t, 3), "hands": hands})
json.dump({"fps": fps, "width": w, "height": h, "frames": frames}, open(out, "w"))
print(len(frames), sum(1 for f in frames if f["hands"]))
