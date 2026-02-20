import streamlit as st
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
import pandas as pd

EMOTION_LABELS = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

st.set_page_config(page_title="Face Emotion Recognition", layout="wide")

# ------------------------------
# Session State
# ------------------------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "enhanced_image" not in st.session_state:
    st.session_state.enhanced_image = None
if "predicted_emotion" not in st.session_state:
    st.session_state.predicted_emotion = None
if "probabilities" not in st.session_state:
    st.session_state.probabilities = None
if "face_coords" not in st.session_state:
    st.session_state.face_coords = None

# ------------------------------
# Load model once at startup
# ------------------------------
@st.cache_resource
def load_model():
    from model import CNN  # your CNN class
    model = CNN()
    model.load_state_dict(
        torch.load("model_cnn_bs32_lr0.001_epoch21.pt", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# ------------------------------
# Preprocessing functions
# ------------------------------
def enhance_image(_image, gamma=0.9):
    """Convert to grayscale, equalize histogram, apply gamma"""
    image = _image.convert("L")
    img_eq = ImageOps.equalize(image)
    img_gamma = Image.fromarray(
        np.uint8(255 * (np.array(img_eq)/255) ** (1/gamma))
    )
    return img_gamma

def preprocess_for_model(_image, resize=(48,48)):
    img_resized = _image.resize(resize)
    img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32).unsqueeze(0)/255.0
    img_tensor = (img_tensor - 0.5)/0.5
    return img_tensor.unsqueeze(0)

# ------------------------------
# Face detection
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(enhanced_image, scale=1.05, min_neighbors=5, expand_ratio=0.2):
    cv_image = np.array(enhanced_image)
    faces = face_cascade.detectMultiScale(cv_image, scaleFactor=scale, minNeighbors=min_neighbors)

    if len(faces) == 0:
        return None, None  # no face detected

    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    x_pad = int(w * expand_ratio)
    y_pad = int(h * expand_ratio)
    x1 = max(x - x_pad, 0)
    y1 = max(y - y_pad, 0)
    x2 = min(x + w + x_pad, cv_image.shape[1])
    y2 = min(y + h + y_pad, cv_image.shape[0])

    face_img = cv_image[y1:y2, x1:x2]
    return Image.fromarray(face_img), (x1, y1, x2-x1, y2-y1)

# ------------------------------
# Upload / Camera
# ------------------------------
image = None
if not st.session_state.prediction_done:
    st.title("Face Emotion Recognition")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    camera_image = st.camera_input("Take a photo")

    if camera_image:
        image = Image.open(camera_image)
    elif uploaded_file:
        image = Image.open(uploaded_file)

# ------------------------------
# Process and Predict
# ------------------------------
if image and not st.session_state.prediction_done:
    st.session_state.original_image = image
    with st.spinner("Processing image..."):
        detected_image, coords = detect_face(image)
        if detected_image is None:
            st.warning("No face detected. Please upload a clear frontal face.")
        else:
            enhanced_image = enhance_image(detected_image)
            input_tensor = preprocess_for_model(enhanced_image)

            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                probabilities = torch.softmax(output, dim=1).numpy()[0]

            # Save to session state
            st.session_state.processed_image = detected_image
            st.session_state.enhanced_image = enhanced_image
            st.session_state.predicted_emotion = EMOTION_LABELS[prediction]
            st.session_state.probabilities = probabilities
            st.session_state.face_coords = coords
            st.session_state.prediction_done = True

            st.rerun()

# ------------------------------
# Show Results (Dashboard Layout)
# ------------------------------
if st.session_state.prediction_done:
    top_left, top_right = st.columns([3, 1])

    with top_left:
        st.markdown(
            f"""
            <h3>Predicted Emotion: 
                <span style='color:{"#26C9FA"}; font-size:45px; font-weight:bold;'>
                    {st.session_state.predicted_emotion}
                </span>
            </h3>
            """,
            unsafe_allow_html=True
        )

    with top_right:
        if st.button("Try Another Image"):
            st.session_state.prediction_done = False
            st.session_state.original_image = None
            st.session_state.processed_image = None
            st.session_state.enhanced_image = None
            st.session_state.predicted_emotion = None
            st.session_state.probabilities = None
            st.session_state.face_coords = None
            st.rerun()

    # Top row: prediction and chart
    col1, col2 = st.columns([3, 1])  # left chart, right prediction

    st.markdown("##### Emotion Probabilities")
    probs_df = pd.DataFrame({
        "Emotion": EMOTION_LABELS,
        "Probability": st.session_state.probabilities
    }).set_index("Emotion")

    st.bar_chart(probs_df.style.bar(color="#26C9FA"))

    if (
        st.session_state.original_image is not None and
        st.session_state.processed_image is not None and
        st.session_state.enhanced_image is not None
    ):
        col1, col2, col3 = st.columns(3)
        
        # Original with face box
        with col1:
            orig_image_with_box = st.session_state.original_image.copy()
            if st.session_state.face_coords is not None:
                x, y, w, h = st.session_state.face_coords
                draw = ImageDraw.Draw(orig_image_with_box)
                draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
            # Resize to same height
            orig_aspect = orig_image_with_box.width / orig_image_with_box.height
            orig_image_resized = orig_image_with_box.resize((int(300 * orig_aspect), 300))
            st.image(orig_image_resized, caption="Original")

        # Detected Face
        with col2:
            det_aspect = st.session_state.processed_image.width / st.session_state.processed_image.height
            det_image_resized = st.session_state.processed_image.resize((int(300 * det_aspect), 300))
            st.image(det_image_resized, caption="Detected Face")

        # Enhanced Face
        with col3:
            enh_aspect = st.session_state.enhanced_image.width / st.session_state.enhanced_image.height
            enh_image_resized = st.session_state.enhanced_image.resize((int(300 * enh_aspect), 300))
            st.image(enh_image_resized, caption="Enhanced Face")

# TODO
# 1. Proprocess val and test data also
# 2. migrate over to vscode
# 3. write read me
# 4. deployment
# 5. UI
# - reset button + reload to result page
# - result bigger