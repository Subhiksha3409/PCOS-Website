import torch
from torchvision import transforms
from PIL import Image
import os
import logging
from model import load_model  # Import the load_model function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the model you want to use for predictions
# Ensure the correct model path is provided here
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')  # Change this if needed
MODEL_FILENAME = 'train_97.94_accuracy_validation_99.57_accuracy_test_99.57_accuracy.pth'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Check if the model path exists for debugging
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
else:
    logging.info(f"Model loaded from {MODEL_PATH}")

# Load the model
model = load_model(MODEL_PATH)

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to the size expected by the model
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensor
])

def predict_image(image_path):
    logging.info(f"Predicting for image: {image_path}")

    # Open the image and apply preprocessing
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Get model outputs
    with torch.no_grad():
        outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_label = torch.max(probabilities, 1)

    # Mapping label indices to actual class labels
    labels = ['Non-PCOS', 'PCOS']
    label = labels[predicted_label.item()]

    logging.info(f"Label: {label}, Confidence: {confidence.item()}")
    return label, confidence.item()

