

import os
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
from Model_Prediction_and_Postprocessing import read_single_image

app = Flask(__name__)
CORS(app, origins='http://localhost:5173')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
GROUND_FOLDER = 'ground'

# Create directories if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GROUND_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['GROUND_FOLDER'] = GROUND_FOLDER

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    # Call post_processing function with input image path and filename
    read_single_image(image_path, image_file.filename)

    processed_image_path = os.path.join(app.config['OUTPUT_FOLDER'], image_file.filename)
    ground_image_path = os.path.join(app.config['GROUND_FOLDER'], image_file.filename)
    
    if not os.path.exists(processed_image_path):
        return 'Processed image not found', 404

    if not os.path.exists(ground_image_path):
        return 'Ground image not found', 404
    
    processed_image_url = url_for('uploaded_file', folder='output', filename=image_file.filename, _external=True)
    ground_image_url = url_for('uploaded_file', folder='ground', filename=image_file.filename, _external=True)
    
    return jsonify({
        'processed_image_url': processed_image_url,
        'ground_image_url': ground_image_url
    })

@app.route('/uploads/<folder>/<filename>')
def uploaded_file(folder, filename):
    if folder not in ['uploads', 'output', 'ground']:
        return 'Invalid folder', 400
    return send_file(os.path.join(app.config[f'{folder.upper()}_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)

