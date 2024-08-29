import os
import time
from flask import Flask, request, render_template, send_from_directory, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path
from ultralytics import YOLO
import tempfile

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure processed folder exists
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load YOLOv8 model
model_path = Path('yolov8n.pt')

if model_path.is_file():
    model = YOLO(model_path)
else:
    print("Error: Could not find model file at", model_path)
    exit()

# Load SAM model
model_path = Path('models/sam_vit_h_4b8939.pth')
if not model_path.is_file():
    print("Error: Could not find model file at", model_path)
    exit()

sam_model = sam_model_registry["vit_h"](checkpoint=str(model_path))
sam_predictor = SamPredictor(sam_model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save uploaded file to a temporary location
        temp_file_path = tempfile.mktemp(suffix=".jpg")
        file.save(temp_file_path)
        
        # Process the image
        processed_image_path = process_image(temp_file_path)
        
        # Delete the temporary uploaded file
        os.remove(temp_file_path)
        
        return render_template('result.html', processed_image=processed_image_path)

def process_image(image_path):
    results = model.predict(source=image_path, conf=0.25, classes=0)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    white_background = np.ones_like(image) * 255
    combined_binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            bbox = box.xyxy.tolist()
            input_box = np.array(bbox)

            sam_predictor.set_image(image)
            masks, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            segmentation_mask = masks[0]
            binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
            combined_binary_mask = np.maximum(combined_binary_mask, binary_mask)

    new_image = white_background * (1 - combined_binary_mask[..., np.newaxis]) + image * combined_binary_mask[..., np.newaxis]
    
    # Save the processed image to a directory
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{int(time.time())}.jpg')
    plt.imsave(processed_image_path, new_image.astype(np.uint8))

    return processed_image_path

@app.route('/processed/<filename>')
def view_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    # This route could be used to manually trigger cleanup
    # Deleting old processed files (e.g., files older than 1 hour)
    now = time.time()
    for filename in os.listdir(app.config['PROCESSED_FOLDER']):
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > 3600:  # 1 hour
            os.remove(file_path)
    return jsonify({'status': 'cleanup complete'})

if __name__ == '__main__':
    app.run(debug=True)
