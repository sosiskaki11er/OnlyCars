import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet(r"D:\archive\yolov4.weights", r"D:\archive\yolov4.cfg")
with open(r"D:\archive\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image
image = cv2.imread("test_image.webp")
height, width, _ = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Run inference
outs = net.forward(output_layers)

# Process detections
car_detected = False
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Check if it's a car and confidence > threshold
        if classes[class_id] == "car" and confidence > 0.5:
            car_detected = True

# Output result
print("1" if car_detected else "0")
