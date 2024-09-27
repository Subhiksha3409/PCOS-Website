import os
import shutil
import torch
from torchvision import transforms
from PIL import Image
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from preprocess import preprocess_images  # Import the preprocessing function

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_path):
    """Load the entire model (architecture + weights)."""
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

def predict_single_image(image_path, model):
    """Predict the label for a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        confidence = confidence.item()
        prediction = 'PCOS' if predicted.item() == 1 else 'Normal'
        
        return prediction, confidence
    
    except Exception as e:
        logging.error(f"Error predicting image {image_path}: {e}")
        return None, None

def create_result_folder(base_dir):
    """Create a numbered result folder if it already exists and return paths for logs and confusion matrix."""
    result_dir = os.path.join(base_dir, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        result_folder = result_dir
    else:
        # Find the next available numbered folder
        i = 1
        while os.path.exists(f"{result_dir}_{i}"):
            i += 1
        result_folder = f"{result_dir}_{i}"
        os.makedirs(result_folder)
    
    # Create directories for logs and confusion matrix
    log_dir = os.path.join(result_folder, 'logs')
    cm_dir = os.path.join(result_folder, 'confusion_matrix')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    
    return result_folder, log_dir, cm_dir

def plot_confusion_matrix(cm, labels, cm_dir, title='Confusion Matrix'):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    # Save the confusion matrix plot
    cm_plot_path = os.path.join(cm_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()

def process_images_in_folder(folder_paths, model, result_folder, log_dir, cm_dir):
    """Process all images in the given folders and save predictions."""
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    pcos_dir = os.path.join(result_folder, 'PCOS')
    normal_dir = os.path.join(result_folder, 'Normal')
    os.makedirs(pcos_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    y_true = []
    y_pred = []

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path).lower()

        for file_name in os.listdir(folder_path):
            if any(file_name.lower().endswith(ext) for ext in supported_extensions):
                image_path = os.path.join(folder_path, file_name)
                result, confidence = predict_single_image(image_path, model)
                if result:
                    dest_dir = pcos_dir if result == 'PCOS' else normal_dir
                    # Use the original folder name and predicted label in the output filename
                    output_file_name = f"{folder_name}_{os.path.splitext(file_name)[0]}_confidence_{confidence:.2f}_predicted_{result}{os.path.splitext(file_name)[1]}"
                    dest_path = os.path.join(dest_dir, output_file_name)
                    shutil.copy2(image_path, dest_path)
                    logging.info(f"Prediction for {file_name}: {result} with confidence {confidence:.2f}. Copied to {dest_dir}")

                    # Collect true and predicted labels for confusion matrix
                    label = 'PCOS' if 'pcos' in folder_name else 'Normal'
                    y_true.append(label)
                    y_pred.append(result)

    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['PCOS', 'Normal'])
    plot_confusion_matrix(cm, labels=['PCOS', 'Normal'], cm_dir=cm_dir)

if __name__ == "__main__":
    # Set the paths to your test input folders
    test_folders = [
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/PCOS',
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/normal'
    ]

    # Set up result folders and logging
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_folder, log_dir, cm_dir = create_result_folder(base_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'predict.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the model
    specific_model_path = os.path.join(base_dir, '..', 'Models', 'train_99.60_accuracy.pth')
    model = load_model(specific_model_path)

    # Process images in the test folders
    process_images_in_folder(test_folders, model, result_folder, log_dir, cm_dir)
    print("Predictions completed. Check the result folder for categorized images and the logs for details.")
