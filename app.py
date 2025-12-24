"""
Flask Backend for Solar Image Segmentation API
===============================================

This is a reference implementation for the segmentation backend.
Run this on your own server (not in Lovable).

Requirements:
    pip install flask flask-cors pillow numpy opencv-python

Usage:
    python app.py
    
    Or with Gunicorn for production:
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

import os
import base64
import zlib
import json
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_WEIGHTS_PATH = os.environ.get('MODEL_WEIGHTS_PATH', './models')

# Placeholder for your actual model loading
# from your_model import CoronalHolesModel, ActiveRegionsModel
# ch_model = CoronalHolesModel.load(f'{MODEL_WEIGHTS_PATH}/ch_model.pth')
# ar_model = ActiveRegionsModel.load(f'{MODEL_WEIGHTS_PATH}/ar_model.pth')


def decode_base64_image(base64_string):
    """Decode base64 image to PIL Image."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def encode_image_to_base64(image, format='PNG'):
    """Encode PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def compress_mask(mask_array):
    """Compress mask array using zlib and encode to base64."""
    mask_bytes = mask_array.tobytes()
    compressed = zlib.compress(mask_bytes)
    return base64.b64encode(compressed).decode('utf-8')


def run_segmentation(image, tasktype, threshold):
    """
    Run the segmentation model on the image.
    
    This is a placeholder - replace with your actual model inference code.
    
    Args:
        image: PIL Image
        tasktype: 'CH' for Coronal Holes or 'AR' for Active Regions
        threshold: 'conservative', 'medium', or 'non-conservative'
    
    Returns:
        tuple: (segmented_image, mask_array, detected_regions)
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Placeholder: Create a dummy segmentation
    # Replace this with your actual model inference
    h, w = img_array.shape[:2]
    
    # Simulated mask (random for demo - replace with actual model output)
    np.random.seed(42)  # For consistent demo results
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Threshold mapping
    threshold_values = {
        'conservative': 0.8,
        'medium': 0.5,
        'non-conservative': 0.3
    }
    thresh = threshold_values.get(threshold, 0.5)
    
    # Simulate detected regions (replace with actual detection)
    detected_regions = []
    num_regions = np.random.randint(3, 8)
    
    for i in range(num_regions):
        # Generate random region position
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        radius = np.random.randint(20, 50)
        
        # Draw on mask
        y, x = np.ogrid[:h, :w]
        region_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
        mask[region_mask] = 255
        
        # Calculate approximate coordinates (simplified)
        longitude = (cx / w) * 360 - 180  # -180 to 180
        latitude = 90 - (cy / h) * 180    # 90 to -90
        area = np.pi * radius ** 2 * 0.001  # Simplified area in Mm²
        
        detected_regions.append({
            'label': i + 1,
            'area_mm2': round(area, 2),
            'longitude': round(longitude, 2),
            'latitude': round(latitude, 2)
        })
    
    # Create segmented output image
    if len(img_array.shape) == 2:
        # Grayscale image
        segmented = np.stack([img_array, img_array, img_array], axis=-1)
    else:
        segmented = img_array.copy()
    
    # Overlay mask with color
    if tasktype == 'CH':
        overlay_color = [0, 255, 255]  # Cyan for Coronal Holes
    else:
        overlay_color = [255, 165, 0]  # Orange for Active Regions
    
    for c in range(3):
        segmented[:, :, c] = np.where(
            mask > 0,
            segmented[:, :, c] * 0.5 + overlay_color[c] * 0.5,
            segmented[:, :, c]
        )
    
    segmented_image = Image.fromarray(segmented.astype(np.uint8))
    
    return segmented_image, mask, detected_regions


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expected JSON body:
    {
        "image": "<base64 encoded image>",
        "threshold": "conservative" | "medium" | "non-conservative",
        "tasktype": "CH" | "AR",
        "instrument": "aia" | "suvi",
        "date": "YYYY-MM-DD" (optional),
        "time": "HH:MM:SS" (optional)
    }
    
    Returns:
    {
        "image": "<base64 encoded segmented image>",
        "mask": "<zlib compressed and base64 encoded mask>",
        "threshold": "...",
        "tasktype": "...",
        "instrument": "...",
        "date": "...",
        "time": "...",
        "stats": {
            "regions": [
                {
                    "label": 1,
                    "area_mm2": 123.45,
                    "longitude": 45.2,
                    "latitude": -12.3
                },
                ...
            ]
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['image', 'threshold', 'tasktype', 'instrument']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters
        image_b64 = data['image']
        threshold = data['threshold']
        tasktype = data['tasktype']
        instrument = data['instrument']
        date = data.get('date', '')
        time = data.get('time', '')
        
        # Validate threshold
        valid_thresholds = ['conservative', 'medium', 'non-conservative']
        if threshold not in valid_thresholds:
            return jsonify({'error': f'Invalid threshold. Must be one of: {valid_thresholds}'}), 400
        
        # Validate tasktype
        valid_tasktypes = ['CH', 'AR']
        if tasktype not in valid_tasktypes:
            return jsonify({'error': f'Invalid tasktype. Must be one of: {valid_tasktypes}'}), 400
        
        # Decode image
        try:
            image = decode_base64_image(image_b64)
        except Exception as e:
            return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
        
        # Run segmentation
        segmented_image, mask, detected_regions = run_segmentation(
            image, tasktype, threshold
        )
        
        # Prepare response
        response = {
            'image': f'data:image/png;base64,{encode_image_to_base64(segmented_image)}',
            'mask': compress_mask(mask),
            'mask_shape': list(mask.shape),
            'threshold': threshold,
            'tasktype': tasktype,
            'instrument': instrument,
            'date': date,
            'time': time,
            'stats': {
                'regions': detected_regions
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'version': '1.0.0'})


@app.route('/thresholds', methods=['GET'])
def get_thresholds():
    """Get available threshold options."""
    return jsonify({
        'thresholds': [
            {'value': 'conservative', 'label': 'Conservative', 'description': 'Fewer, more certain regions'},
            {'value': 'medium', 'label': 'Medium', 'description': 'Balanced detection'},
            {'value': 'non-conservative', 'label': 'Non-conservative', 'description': 'More, possibly less certain regions'}
        ]
    })


if __name__ == '__main__':
    print("Starting Solar Segmentation API...")
    print("Endpoints:")
    print("  POST /predict - Run segmentation on an image")
    print("  GET /health - Health check")
    print("  GET /thresholds - Get available threshold options")
    app.run(host='0.0.0.0', port=5000, debug=True)
