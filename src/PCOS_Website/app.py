import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import logging
from PIL import Image  # Ensure PIL is imported

from predict import predict_image  # Import the function from predict.py

# Rest of your code...


# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to validate and resize image if necessary
def validate_image_size(image_path, max_size=(1024, 1024)):
    with Image.open(image_path) as img:
        if img.size > max_size:
            img.thumbnail(max_size)
            img.save(image_path)

# Route for the homepage (upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Updated route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Validate and resize the image if necessary
            validate_image_size(filepath)

            # Make a prediction on the uploaded image
            label, confidence = predict_image(filepath)

            return render_template('result.html', label=label, confidence=confidence, image_filename=filename)

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return "Error processing the image", 500

    return "Invalid file", 400

# Route to serve the result page
@app.route('/result/<filename>')
def display_result(filename):
    return redirect(url_for('static', filename=os.path.join('uploads', filename)), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
