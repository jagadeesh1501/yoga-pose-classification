"""
STEP 13: APPLICATION DEVELOPMENT (Streamlit)
Yoga Pose Classification using MediaPipe Pose + ANN

Run with: streamlit run step13_streamlit_app.py
"""

import numpy as np
import cv2
from PIL import Image
import streamlit as st

from step12a_mediapipe_integration import YogaPosePredictor

st.set_page_config(page_title="Yoga Pose Classifier", layout="centered")


@st.cache_resource
def load_predictor():
    return YogaPosePredictor()


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    predictor = load_predictor()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict"])

    if page == "Home":
        st.title("🧘 Yoga Pose Classification")
        st.markdown("""
        ### Project Description
        This app classifies a yoga pose from an image using **MediaPipe Pose**
        for landmark extraction and an **Artificial Neural Network (ANN)**
        for classification.

        **Pipeline**: Image → MediaPipe Pose (33 landmarks) → Joint angle
        feature engineering → Scaling → ANN → Predicted pose + confidence

        **Supported poses**: Downward Dog, Tree, Warrior 1, Goddess,
        Mountain, Warrior 2

        ### How to use
        1. Go to the **Predict** page from the sidebar.
        2. Either upload an image or use your webcam.
        3. View the predicted pose and confidence score.

        **Tip**: Make sure your full body is visible in the frame for
        the most accurate landmark detection.
        """)

    elif page == "Predict":
        st.title("Predict Your Yoga Pose")

        tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Webcam"])

        with tab1:
            uploaded_file = st.file_uploader("Upload a yoga pose image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                bgr_img = pil_to_bgr(pil_img)

                label, confidence, annotated = predictor.predict(bgr_img)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb, caption="Detected Landmarks", use_container_width=True)

                if label is None:
                    st.warning("No pose detected. Try an image with the full body clearly visible.")
                else:
                    st.success(f"**Predicted Pose:** {label}")
                    st.metric("Confidence Score", f"{confidence * 100:.1f}%")

        with tab2:
            camera_image = st.camera_input("Take a photo")
            if camera_image is not None:
                pil_img = Image.open(camera_image)
                bgr_img = pil_to_bgr(pil_img)

                label, confidence, annotated = predictor.predict(bgr_img)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb, caption="Detected Landmarks", use_container_width=True)

                if label is None:
                    st.warning("No pose detected. Try again with your full body clearly visible.")
                else:
                    st.success(f"**Predicted Pose:** {label}")
                    st.metric("Confidence Score", f"{confidence * 100:.1f}%")

        st.markdown("---")
        st.caption("User Instructions: Stand a few feet from the camera so your "
                    "full body (head to feet) is visible for best accuracy.")


if __name__ == "__main__":
    main()