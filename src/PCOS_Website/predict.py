import torch
from torchvision import transforms
from PIL import Image
import os
from model import SimpleCNN  # Import the SimpleCNN architecture

# Load your trained CNN model (update the path to your saved model file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'pcos_classification_model.pth')

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN()  # Ensure this matches the architecture
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load model weights
model.to(device)  # Move the model to the appropriate device (CPU or GPU)
model.eval()  # Set model to evaluation mode

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to the size expected by the model
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensor
])

def predict_image(image_path):
    # Open the image and apply preprocessing
    image = Image.open(image_path).convert('RGB')  # Ensure it's in RGB format
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)  # Move the image to the same device as the model

    # Get model outputs
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(image)
    
    probabilities = torch.softmax(outputs, dim=1)  # Get probabilities
    confidence, predicted_label = torch.max(probabilities, 1)  # Get confidence and predicted label

    # Mapping label indices to actual class labels
    labels = ['Non-PCOS', 'PCOS']
    label = labels[predicted_label.item()]  # Convert index to label

    print(f"Label: {label}, Confidence: {confidence.item()}")  # Debug log

    return label, confidence.item()


