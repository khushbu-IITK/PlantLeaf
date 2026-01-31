import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon="🥔",
    layout="centered",
)

# Must match your model's output order (Dense(3))
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy",
]

# Must match your model's expected input (from your summary)
IMG_SIZE = (256, 256)

# If you trained with pixel scaling (common), keep True.
# If your model includes a Rescaling(1./255) layer, set this to False.
RESCALE_255 = False
# -------------------------------------------


@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    # compile=False avoids issues if custom metrics/loss were used
    return  model # tf.keras.models.load_model("model.h5", compile=False)


def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image -> (1, 256, 256, 3) float32 tensor-like numpy array."""
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)

    arr = np.array(pil_img).astype(np.float32)

    if RESCALE_255:
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(model, pil_img: Image.Image):
    x = preprocess_pil(pil_img)
    preds = model.predict(x)

    preds = np.array(preds)
    # Expected shape (1, 3)
    if preds.ndim == 2:
        probs = preds[0]
    else:
        probs = preds.flatten()

    if len(probs) != len(CLASS_NAMES):
        raise ValueError(
            f"Model output has {len(probs)} classes, but CLASS_NAMES has {len(CLASS_NAMES)}."
        )

    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(np.max(probs) * 100.0)

    return label, round(confidence, 2), probs


# ------------------ UI ------------------
st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload a potato leaf image and this app will predict the disease class.")

with st.expander("Settings / Debug", expanded=False):
    st.write(f"**Expected input size:** {IMG_SIZE[0]} x {IMG_SIZE[1]} x 3")
   # st.write(f"**Rescale /255 enabled:** {RESCALE_255}")
    st.write(f"**Classes ({len(CLASS_NAMES)}):** {CLASS_NAMES}")

# Load model once
model = load_model()

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Please upload a potato leaf image to get a prediction.")
    st.stop()

# Read uploaded image
try:
    pil_img = Image.open(uploaded)
except Exception:
    st.error("That file doesn't look like a valid image. Please upload JPG/PNG.")
    st.stop()

st.image(pil_img, caption="Uploaded image", use_container_width=True)

# Predict
with st.spinner("Predicting..."):
    try:
        label, conf, probs = predict_image(model, pil_img)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

st.subheader("Result")
st.metric("Predicted class", label, f"{conf}% confidence")

st.subheader("Class probabilities")
prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
st.bar_chart(prob_dict)

#st.caption("Tip: If confidence looks weird, check that RESCALE_255 matches your training preprocessing.")

