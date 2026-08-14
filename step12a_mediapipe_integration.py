"""
STEP 12a: MEDIAPIPE INTEGRATION
Yoga Pose Classification using MediaPipe Pose + ANN
"""

import os
import math
import cv2
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

FINAL_DIR = os.path.join(os.getcwd(), "final_model")


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


class YogaPosePredictor:
    def __init__(self, model_dir=FINAL_DIR):
        self.model = tf.keras.models.load_model(os.path.join(model_dir, "yoga_pose_model.keras"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def extract_features(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        annotated = image_bgr.copy()

        if not results.pose_landmarks:
            return None, annotated, False

        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        lm = results.pose_landmarks.landmark
        coords = {}
        for i, point in enumerate(lm):
            coords[f"x{i}"] = point.x
            coords[f"y{i}"] = point.y
            coords[f"z{i}"] = point.z
            coords[f"v{i}"] = point.visibility

        def pt(idx):
            return (lm[idx].x, lm[idx].y)

        L_SHOULDER, R_SHOULDER = 11, 12
        L_ELBOW, R_ELBOW = 13, 14
        L_WRIST, R_WRIST = 15, 16
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28

        angles = {
            "left_elbow_angle": calculate_angle(pt(L_SHOULDER), pt(L_ELBOW), pt(L_WRIST)),
            "right_elbow_angle": calculate_angle(pt(R_SHOULDER), pt(R_ELBOW), pt(R_WRIST)),
            "left_shoulder_angle": calculate_angle(pt(L_ELBOW), pt(L_SHOULDER), pt(L_HIP)),
            "right_shoulder_angle": calculate_angle(pt(R_ELBOW), pt(R_SHOULDER), pt(R_HIP)),
            "left_hip_angle": calculate_angle(pt(L_SHOULDER), pt(L_HIP), pt(L_KNEE)),
            "right_hip_angle": calculate_angle(pt(R_SHOULDER), pt(R_HIP), pt(R_KNEE)),
            "left_knee_angle": calculate_angle(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE)),
            "right_knee_angle": calculate_angle(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE)),
        }

        features = {**coords, **angles}
        return features, annotated, True

    def predict(self, image_bgr):
        features, annotated, found = self.extract_features(image_bgr)
        if not found:
            return None, None, annotated

        X = np.array([[features[c] for c in self.feature_cols]])
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict(X_scaled, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        label = self.encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        return label, confidence, annotated