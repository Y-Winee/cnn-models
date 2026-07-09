import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as inception_preprocess_input,
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.models import load_model


def _patch_dense_for_keras3():
    try:
        from keras.src.layers.core.dense import Dense as KerasDense
        from tensorflow.keras.layers import Dense as TFDense

        if getattr(TFDense, "_copilot_patched", False):
            return

        original_init = TFDense.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            return original_init(self, *args, **kwargs)

        TFDense.__init__ = patched_init
        KerasDense.__init__ = patched_init
        TFDense._copilot_patched = True
        KerasDense._copilot_patched = True
    except Exception:
        pass


_patch_dense_for_keras3()

ROOT = Path(__file__).resolve().parent
CLASSES_PATH = ROOT / "classes.json"

MODEL_OPTIONS = [
    {
        "name": "Scratch CNN",
        "file": "cnn_accuracy_84.keras",
        "kind": "scratch",
        "size": (220, 220),
        "description": "CNN trained from scratch (84% accuracy)",
    },
    {
        "name": "Feature Extracted CNN",
        "file": "feature_extracted_cnn_accuracy_94.keras",
        "kind": "vgg16",
        "size": (224, 224),
        "description": "VGG16-based feature extraction model (94% accuracy)",
    },
    {
        "name": "Transfer Learning",
        "file": "transfer_learning_accuracy_93.keras",
        "kind": "inception",
        "size": (224, 224),
        "description": "InceptionResNetV2 transfer learning model (93% accuracy)",
    },
]


@st.cache_resource
def load_classes():
    return json.loads(CLASSES_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_selected_model(model_name: str):
    selected = next(item for item in MODEL_OPTIONS if item["name"] == model_name)
    model_path = ROOT / selected["file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if selected["kind"] == "scratch":
            model = load_model(model_path, compile=False)
        elif selected["kind"] == "vgg16":
            model = load_model(
                model_path,
                compile=False,
                custom_objects={"preprocess_input": vgg16_preprocess_input},
            )
        else:
            model = load_model(
                model_path,
                compile=False,
                custom_objects={"preprocess_input": inception_preprocess_input},
            )
    except Exception as exc:
        raise RuntimeError(f"Unable to load {selected['name']}: {exc}") from exc

    return model, selected


def preprocess_image(image: Image.Image, selected: dict):
    image = image.convert("RGB")
    image = image.resize(selected["size"])
    st.write(f"{image.size}")
    image_array = np.asarray(image, dtype=np.float32)

    if selected["kind"] == "scratch":
        image_array = image_array / 255.0
        return np.expand_dims(image_array, axis=0)

    image_array = np.expand_dims(image_array, axis=0)
    if selected["kind"] == "vgg16":
        return vgg16_preprocess_input(image_array)

    return inception_preprocess_input(image_array)


def predict_image(image: Image.Image, model, selected: dict, classes):
    processed = preprocess_image(image, selected)
    predictions = model.predict(processed, verbose=0)[0]
    index = int(np.argmax(predictions))
    return classes[index], float(predictions[index] * 100), predictions


st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Flower Image Classifier")
st.caption("Choose a trained model from the dropdown and upload a flower image to classify it.")

with st.sidebar:
    st.header("Model selection")
    model_name = st.selectbox(
        "Pick a model",
        options=[item["name"] for item in MODEL_OPTIONS],
        index=0,
    )
    selected = next(item for item in MODEL_OPTIONS if item["name"] == model_name)
    st.info(selected["description"])
    st.caption(f"Expected input size: {selected['size'][0]}x{selected['size'][1]}")

uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    classes = load_classes()

    try:
        with st.spinner("Running prediction..."):
            model, selected_model = load_selected_model(model_name)
            predicted_class, confidence, probabilities = predict_image(
                image, model, selected_model, classes
            )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Predicted class: {predicted_class}")
    st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader("Class probabilities")
    for class_name, prob in sorted(
        zip(classes, probabilities), key=lambda item: item[1], reverse=True
    ):
        st.write(f"- {class_name}: {prob * 100:.2f}%")
else:
    st.info("Upload an image to see the prediction for the selected model.")
