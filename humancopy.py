# humancopy.py - Python 3.11 + MediaPipe 0.10.31

import cv2
import mediapipe as mp
import time
from collections import deque
import copy

# ------------------ SETTINGS ------------------
DELAY_SECONDS = 0.7
MAX_BUFFER_SIZE = 120
# ----------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pose_buffer = deque(maxlen=MAX_BUFFER_SIZE)
prev_delayed_pose = None


def smooth_landmarks(prev, curr, alpha=0.7):
    if prev is None:
        return curr

    for i in range(len(curr.landmark)):
        curr.landmark[i].x = (
            alpha * curr.landmark[i].x + (1 - alpha) * prev.landmark[i].x
        )
        curr.landmark[i].y = (
            alpha * curr.landmark[i].y + (1 - alpha) * prev.landmark[i].y
        )
        curr.landmark[i].z = (
            alpha * curr.landmark[i].z + (1 - alpha) * prev.landmark[i].z
        )
    return curr


def draw_skeleton(image, landmarks, color=(0, 255, 255), mirror=True):
    h, w, _ = image.shape
    points = []

    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)

        if mirror:
            x = w - x

        points.append((x, y))

    for p in points:
        cv2.circle(image, p, 4, color, -1)

    for start, end in mp_pose.POSE_CONNECTIONS:
        cv2.line(image, points[start], points[end], color, 2)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    # ✅ WINDOW SIZE FIX (ADD THIS)
    cv2.namedWindow("Human Motion Imitation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Human Motion Imitation", 1280, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        current_time = time.time()

        if results.pose_landmarks:
            pose_buffer.append(
                (current_time, copy.deepcopy(results.pose_landmarks))
            )

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        delayed_pose = None
        for t, p in pose_buffer:
            if current_time - t >= DELAY_SECONDS:
                delayed_pose = p
                break

        if delayed_pose:
         delayed_pose = smooth_landmarks(prev_delayed_pose, delayed_pose)
         prev_delayed_pose = copy.deepcopy(delayed_pose)

         overlay = frame.copy()
         draw_skeleton(overlay, delayed_pose, color=(0, 255, 255))
         frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        cv2.putText(
            frame,
            f"Human-like Delay: {DELAY_SECONDS}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Human Motion Imitation", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
