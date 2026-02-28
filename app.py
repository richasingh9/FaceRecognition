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
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
col1, col2 = st.columns(2)

fps_display = col1.empty()
face_count_display = col2.empty()

if run:
    cap = cv2.VideoCapture(0)
    prev_time = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
        img_tensor = img_tensor.unsqueeze(0)

        start_time = time.time()

        with torch.no_grad():
            outputs = model(img_tensor)

        end_time = time.time()
        inference_time = end_time - start_time

        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        face_count = 0

        for box, score in zip(boxes, scores):
            if score > 0.7:
                face_count += 1
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        fps_display.metric("FPS", int(fps))
        face_count_display.metric("Detected Faces", face_count)

    cap.release()