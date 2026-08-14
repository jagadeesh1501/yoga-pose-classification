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


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# MODEL DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FINAL_DIR = os.path.join(BASE_DIR, "final_model")


# ============================================================
# ANGLE CALCULATION
# ============================================================

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    Point b is the vertex.
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(radians * 180.0 / math.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


# ============================================================
# YOGA POSE PREDICTOR
# ============================================================

class YogaPosePredictor:

    def __init__(self, model_dir=FINAL_DIR):

        # File paths
        model_path = os.path.join(
            model_dir,
            "yoga_pose_model.keras"
        )

        scaler_path = os.path.join(
            model_dir,
            "scaler.pkl"
        )

        encoder_path = os.path.join(
            model_dir,
            "label_encoder.pkl"
        )

        feature_path = os.path.join(
            model_dir,
            "feature_cols.pkl"
        )

        # Check whether all required files exist
        required_files = [
            model_path,
            scaler_path,
            encoder_path,
            feature_path
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Required file not found:\n{file_path}"
                )

        # Load ANN model
        print("Loading Yoga Pose ANN model...")

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print("Model loaded successfully!")

        # Load preprocessing files
        self.scaler = joblib.load(scaler_path)

        self.encoder = joblib.load(encoder_path)

        self.feature_cols = joblib.load(feature_path)

        # Initialize MediaPipe Pose
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(self, image_bgr):

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        # Process image using MediaPipe
        results = self.pose.process(image_rgb)

        # Copy image for drawing landmarks
        annotated = image_bgr.copy()

        # If no person/pose detected
        if not results.pose_landmarks:

            return None, annotated, False

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Get landmarks
        lm = results.pose_landmarks.landmark


        # ====================================================
        # EXTRACT LANDMARK COORDINATES
        # ====================================================

        coords = {}

        for i, point in enumerate(lm):

            coords[f"x{i}"] = point.x
            coords[f"y{i}"] = point.y
            coords[f"z{i}"] = point.z
            coords[f"v{i}"] = point.visibility


        # Helper function
        def pt(index):

            return (
                lm[index].x,
                lm[index].y
            )


        # ====================================================
        # MEDIAPIPE LANDMARK INDEXES
        # ====================================================

        L_SHOULDER = 11
        R_SHOULDER = 12

        L_ELBOW = 13
        R_ELBOW = 14

        L_WRIST = 15
        R_WRIST = 16

        L_HIP = 23
        R_HIP = 24

        L_KNEE = 25
        R_KNEE = 26

        L_ANKLE = 27
        R_ANKLE = 28


        # ====================================================
        # CALCULATE BODY ANGLES
        # ====================================================

        angles = {

            "left_elbow_angle":

                calculate_angle(
                    pt(L_SHOULDER),
                    pt(L_ELBOW),
                    pt(L_WRIST)
                ),

            "right_elbow_angle":

                calculate_angle(
                    pt(R_SHOULDER),
                    pt(R_ELBOW),
                    pt(R_WRIST)
                ),

            "left_shoulder_angle":

                calculate_angle(
                    pt(L_ELBOW),
                    pt(L_SHOULDER),
                    pt(L_HIP)
                ),

            "right_shoulder_angle":

                calculate_angle(
                    pt(R_ELBOW),
                    pt(R_SHOULDER),
                    pt(R_HIP)
                ),

            "left_hip_angle":

                calculate_angle(
                    pt(L_SHOULDER),
                    pt(L_HIP),
                    pt(L_KNEE)
                ),

            "right_hip_angle":

                calculate_angle(
                    pt(R_SHOULDER),
                    pt(R_HIP),
                    pt(R_KNEE)
                ),

            "left_knee_angle":

                calculate_angle(
                    pt(L_HIP),
                    pt(L_KNEE),
                    pt(L_ANKLE)
                ),

            "right_knee_angle":

                calculate_angle(
                    pt(R_HIP),
                    pt(R_KNEE),
                    pt(R_ANKLE)
                )
        }


        # ====================================================
        # COMBINE FEATURES
        # ====================================================

        features = {
            **coords,
            **angles
        }

        return features, annotated, True


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, image_bgr):

        # Extract features
        features, annotated, found = self.extract_features(
            image_bgr
        )


        # If pose is not detected
        if not found:

            return None, None, annotated


        # Arrange features in the exact order
        # used during model training
        X = np.array([
            [
                features[column]
                for column in self.feature_cols
            ]
        ])


        # Apply scaler
        X_scaled = self.scaler.transform(X)


        # ANN prediction
        probabilities = self.model.predict(
            X_scaled,
            verbose=0
        )[0]


        # Get highest probability
        prediction_index = int(
            np.argmax(probabilities)
        )


        # Convert encoded prediction to pose name
        label = self.encoder.inverse_transform(
            [prediction_index]
        )[0]


        # Confidence
        confidence = float(
            probabilities[prediction_index]
        )


        return label, confidence, annotated


    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self):

        if hasattr(self, "pose"):

            self.pose.close()