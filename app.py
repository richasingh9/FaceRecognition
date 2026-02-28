import cv2
import torch
import torchvision
import time
import numpy as np

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# Load Pretrained Face Detection Model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model = model.to(device)
model.eval()

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
    img_tensor = img_tensor.unsqueeze(0)

    # Inference Timing
    start_time = time.time()

    with torch.no_grad():
        outputs = model(img_tensor)

    end_time = time.time()
    inference_time = end_time - start_time

    # Process Detections
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    for box, score in zip(boxes, scores):
        if score > 0.7:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show Inference Time
    cv2.putText(frame, f"Inference Time: {inference_time:.3f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    cv2.imshow("Real-Time Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()


