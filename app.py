
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter

MODEL_PATH = os.path.join("models", "digit_model.keras")

st.set_page_config(
    page_title="Smart Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    """Convert an uploaded image into a 28x28 MNIST-style image."""
    image = ImageOps.exif_transpose(image).convert("L")

    # Improve contrast and remove light background variations.
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.MedianFilter(size=3))

    # MNIST images have dark backgrounds and bright digits.
    # Invert when the background is predominantly bright.
    pixels = np.array(image)
    if np.mean(pixels) > 127:
        image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    pixels = np.array(image).astype("float32") / 255.0
    pixels = pixels.reshape(1, 28, 28, 1)

    return pixels


st.title("🔢 Smart Digit Recognizer")
st.write(
    "Recognize handwritten digits from 0 to 9 "
    "using a Convolutional Neural Network."
)

st.info(
    "For best results, upload a clear image containing "
    "one large handwritten digit on a plain background."
)

model = load_model()

if model is None:
    st.warning(
        "The trained model was not found. "
        "First run: python train_model.py"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded image")
    st.image(image, width=250)

    processed = preprocess_image(image)

    if st.button("Recognize Digit", type="primary"):
        probabilities = model.predict(processed, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])

        st.success(f"Predicted Digit: {prediction}")
        st.metric(
            "Model confidence",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Prediction probabilities")

        chart_data = {
            str(digit): float(probability)
            for digit, probability in enumerate(probabilities)
        }
        st.bar_chart(chart_data)

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

st.divider()
st.caption(
    "Built with Python, TensorFlow, Keras, Streamlit, and MNIST."
)
