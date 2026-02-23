import streamlit as st
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

EMOTION_LABELS = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

# ------------------------------
# Styling
# ------------------------------
st.markdown(
    """
    <style>
    /* Constrain the entire page content */
    .app-container {
        max-width: 600px;  /* change this to your preferred width */
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

#
# ------------------------------
# Functions
# ------------------------------
# Load model once at startup
@st.cache_resource
def load_model():
    from model import CNN  # your CNN class
    model = CNN()
    model.load_state_dict(
        torch.load("model_cnn_bs32_lr0.001_epoch21.pt", map_location="cpu")
    )
    model.eval()
    return model

# Image enhancement: grayscale, histogram equalization, gamma correction
def enhance_image(_image, gamma=0.9):
    """Convert to grayscale, equalize histogram, apply gamma"""
    image = _image.convert("L")
    img_eq = ImageOps.equalize(image)
    img_gamma = Image.fromarray(
        np.uint8(255 * (np.array(img_eq)/255) ** (1/gamma))
    )
    return img_gamma

# Preprocess for model: resize, normalize
def preprocess_for_model(_image, resize=(48,48)):
    img_resized = _image.resize(resize)
    img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32).unsqueeze(0)/255.0
    img_tensor = (img_tensor - 0.5)/0.5
    return img_tensor.unsqueeze(0)

# Face detection using OpenCV Haar cascades
def detect_face(enhanced_image, scale=1.05, min_neighbors=5, expand_ratio=0.2):
    cv_image = np.array(enhanced_image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
# Main App
# ------------------------------
model = load_model()
st.markdown('<div class="app-container">', unsafe_allow_html=True)
st.set_page_config(page_title="Face Emotion Recognition", layout="centered")
st.title("Face Emotion Recognition")

# Upload / Camera
image = None
if not st.session_state.prediction_done:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    camera_image = st.camera_input("Take a photo")

    if camera_image:
        image = Image.open(camera_image)
    elif uploaded_file:
        image = Image.open(uploaded_file)

# Process and Predict
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

# Show Results (Dashboard Layout)
if st.session_state.prediction_done:
    row1_col1, row1_col2 = st.columns([1, 1])

    # Pie chart
    with row1_col2:
        probs = st.session_state.probabilities  # your probabilities array
        predicted_emotion = st.session_state.predicted_emotion

        # Prepare DataFrame
        probs_df = pd.DataFrame({
            "Emotion": EMOTION_LABELS,
            "Probability": probs
        })

        # Create a color map from Viridis
        colors = px.colors.sequential.Viridis_r
        # Map each emotion to a color proportionally along Viridis scale
        num_colors = len(EMOTION_LABELS)
        emotion_color_map = {emotion: colors[int(i*(len(colors)-1)/(num_colors-1))] 
                            for i, emotion in enumerate(EMOTION_LABELS)}

        # Get the color of the highest probability emotion
        max_emotion = probs_df.loc[probs_df['Probability'].idxmax(), 'Emotion']
        color_for_text = emotion_color_map[max_emotion]

        # Plot pie chart
        fig = px.pie(
            probs_df,
            names="Emotion",
            values="Probability",
            color="Emotion",
            color_discrete_map=emotion_color_map,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=20,b=20,l=20,r=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Predicted emotion + button
    with row1_col1:
        st.markdown(
            f"""
            <h3 style="margin-top: 5px;">Predicted Emotion: 
                <span style='color:{color_for_text}; font-size:45px; font-weight:bold;'>
                    {st.session_state.predicted_emotion}
                </span>
            </h3>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)  # spacing
        if st.button("Try Another Image"):
            for key in ["prediction_done","original_image","processed_image","enhanced_image",
                        "predicted_emotion","probabilities","face_coords"]:
                st.session_state[key] = None
            # Stop the current script to refresh the page
            st.rerun()

if (
    st.session_state.original_image is not None and
    st.session_state.processed_image is not None and
    st.session_state.enhanced_image is not None
):
    st.markdown(
        "<p style='text-align: center; font-size: 24px;'>Image Preprocessing</p>",
        unsafe_allow_html=True
    )
    
    img_col1, img_col2, img_col3 = st.columns([1,1,1], gap="medium")

    def resize_to_height(image, target_height):
        aspect_ratio = image.width / image.height
        new_width = int(target_height * aspect_ratio)
        return image.resize((new_width, target_height))

    # Original with face box
    with img_col1:
        orig_image_with_box = st.session_state.original_image.copy()
        if st.session_state.face_coords is not None:
            x, y, w, h = st.session_state.face_coords
            draw = ImageDraw.Draw(orig_image_with_box)
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
        st.image(resize_to_height(orig_image_with_box, 200), caption="Original Image")

    # Detected face
    with img_col2:
        st.image(resize_to_height(st.session_state.processed_image, 200), caption="Detected Face Image")

    # Enhanced face
    with img_col3:
        st.image(resize_to_height(st.session_state.enhanced_image, 200), caption="Preprocessed Image")

st.markdown('</div>', unsafe_allow_html=True)

# TODO
# 1. Proprocess val and test data also
# 2. migrate over to vscode
# 3. write read me
# 4. deployment
# 5. UI
# - reset button + reload to result page
# - result bigger