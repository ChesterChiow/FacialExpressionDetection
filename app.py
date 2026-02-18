import streamlit as st
import torch
from PIL import Image, ImageOps
import numpy as np
import cv2

st.title("Face Emotion Recognition")

# ------------------------------
# Load model once at startup
# ------------------------------
@st.cache_resource
def load_model():
    from model.cnn import CNN
    model = CNN()
    model.load_state_dict(
        torch.load("checkpoints/model_cnn_bs64_lr0.001_epoch52.pt", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# ------------------------------
# Preprocessing functions
# ------------------------------
def enhance_image(_image, gamma=0.9):
    image = _image.convert("L")
    img_eq = ImageOps.equalize(image)
    img_gamma = Image.fromarray(np.uint8(255 * (np.array(img_eq)/255) ** (1/gamma)))
    return img_gamma

def preprocess_for_model(_image, resize=(48,48)):
    img_resized = _image.resize(resize)
    img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32).unsqueeze(0)/255.0
    img_tensor = (img_tensor - 0.5)/0.5
    input_tensor = img_tensor.unsqueeze(0)
    return input_tensor

# ------------------------------
# Face detection
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(enhanced_image):
    cv_image = np.array(enhanced_image)
    faces = face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return enhanced_image
    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    face_img = cv_image[y:y+h, x:x+w]
    return Image.fromarray(face_img)

# ------------------------------
# Upload or camera input (mutually exclusive)
# ------------------------------

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
camera_image = st.camera_input("Take a photo")

# Use **camera image first if available**, otherwise uploaded file
image = None
if camera_image:
    image = Image.open(camera_image)
elif uploaded_file:
    image = Image.open(uploaded_file)

# ------------------------------
# 5️⃣ Process and predict
# ------------------------------
if image:
    with st.spinner("Enhancing image, detecting face, and predicting..."):
        enhanced_image = enhance_image(image)
        face_image = detect_face(enhanced_image)

        # Check if face detected
        if face_image.size == enhanced_image.size:
            st.warning("No face detected. Please retake/upload a clear frontal face.")
        else:
            input_tensor = preprocess_for_model(face_image)
            st.image(face_image, caption="Detected Face + Enhanced")

            # Prediction
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                probabilities = torch.softmax(output, dim=1).numpy()[0]

            emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

            st.subheader("Emotion Probabilities")
            st.bar_chart({label: float(prob) for label, prob in zip(emotion_labels, probabilities)})

            predicted_emotion = emotion_labels[prediction]
            st.success(f"Prediction: {predicted_emotion}")