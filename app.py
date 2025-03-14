from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import time

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # YOLOv8 nano (use "yolov8s.pt" for more accuracy)

@app.route('/')
def home():
    return render_template("index.html")  # Serve the frontend page

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Perform Object Detection with timing
    start_time = time.time()
    results = model(file_path)
    end_time = time.time()

    processing_time = round((end_time - start_time) * 1000, 2)  # Convert to ms

    detected_objects = []
    img = cv2.imread(file_path)

    for result in results:
        inference_time = result.speed['inference']  # Inference time per image
        for box in result.boxes:
            cls = int(box.cls[0])  # Get class index
            obj_name = model.names[cls]  # Get object name
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            detected_objects.append({
                "name": obj_name,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, obj_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save processed image
    processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
    cv2.imwrite(processed_path, img)

    return jsonify({
        "objects": detected_objects,
        "image_url": f"/processed/{file.filename}",
        "detection_speed": processing_time,
        "inference_time": inference_time
    })

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed/<filename>')
def get_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
