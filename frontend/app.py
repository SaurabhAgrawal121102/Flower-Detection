import streamlit as st
import requests
from PIL import Image
import io


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Flower Recognition",
    page_icon="🌸",
    layout="centered"
)


# =========================================================
# Title
# =========================================================

st.title("🌸 Flower Recognition")
st.write("Upload a flower image and let EfficientNet-B0 classify it.")


# =========================================================
# FastAPI Backend URL
# =========================================================

API_URL = "https://flower-detection-api-izfn.onrender.com/predict"


# =========================================================
# Image Upload
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png", "webp"]
)


# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Open image
    # -----------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    # -----------------------------------------------------
    # Display image
    # -----------------------------------------------------

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Your uploaded image",
        width="stretch"
    )

    # -----------------------------------------------------
    # Predict button
    # -----------------------------------------------------

    if st.button("🔍 Predict Flower"):

        with st.spinner("Analyzing image..."):

            try:

                # Reset file position
                uploaded_file.seek(0)

                # Send image to FastAPI
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            uploaded_file.type
                        )
                    }
                )

                # -------------------------------------------------
                # Successful response
                # -------------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    st.success(
                        f"🌸 Prediction: {prediction}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                # -------------------------------------------------
                # Backend error
                # -------------------------------------------------

                else:

                    st.error(
                        f"Backend Error: {response.status_code}"
                    )

                    st.write(
                        response.text
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to FastAPI backend."
                )

                st.info(
                    "Make sure your FastAPI server is running."
                )