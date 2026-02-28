import streamlit as st
import cv2
import torch
import torchvision
import time
import numpy as np

st.set_page_config(layout="wide")

st.title("🚀 Real-Time Smart Classroom Dashboard")
st.write("GPU Accelerated Face Detection with Live Analytics")

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write("### Device Info")
st.sidebar.write(f"Using Device: {device}")

# Load Model
@st.cache_resource
def load_model():
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Start Camera
from PIL import Image

st.subheader("📂 Upload Image for Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    img_tensor = torchvision.transforms.functional.to_tensor(img)
    img_tensor = img_tensor.unsqueeze(0)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(img_tensor)

    end_time = time.time()
    inference_time = end_time - start_time

    boxes = outputs[0]['boxes'].numpy()
    scores = outputs[0]['scores'].numpy()

    face_count = 0

    for box, score in zip(boxes, scores):
        if score > 0.7:
            face_count += 1
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(img, caption=f"Detected Objects: {face_count}")
    st.write(f"Inference Time: {inference_time:.3f} seconds")
