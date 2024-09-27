import os
import torch
from torch.utils.data import DataLoader
from data_loader import PCOSDataset, transform
from train import train_model
from validate import validate_model
from test import test_model

def main():
    # Define paths
    train_dirs = [
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/train/PCOS',
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/train/normal'
    ]
    val_dirs = [
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/validation/normal',
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/validation/PCOS'
    ]
    test_dirs = [
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/PCOS',
        '/Users/subhikshaswaminathan/Desktop/Everything/src/Dataset/test/normal'
    ]

    try:
        # Create DataLoaders
        train_datasets = [PCOSDataset(image_folder=folder, transform=transform) for folder in train_dirs]
        train_loader = DataLoader(
            dataset=torch.utils.data.ConcatDataset(train_datasets),
            batch_size=32,
            shuffle=True
        )

        val_datasets = [PCOSDataset(image_folder=folder, transform=transform) for folder in val_dirs]
        val_loader = DataLoader(
            dataset=torch.utils.data.ConcatDataset(val_datasets),
            batch_size=32,
            shuffle=False
        )

        test_datasets = [PCOSDataset(image_folder=folder, transform=transform) for folder in test_dirs]
        test_loader = DataLoader(
            dataset=torch.utils.data.ConcatDataset(test_datasets),
            batch_size=32,
            shuffle=False
        )

        # Train model
        train_accuracy, trained_model = train_model(train_loader)
        
        # Validate model
        validate_accuracy = validate_model(val_loader, trained_model)

        # Test model
        test_accuracy = test_model(test_loader, trained_model)

        # Prepare model save path
        accuracy_labels = []
        
        if train_accuracy >= 90:
            accuracy_labels.append(f'train_{train_accuracy:.2f}_accuracy')

        if validate_accuracy >= 90:
            accuracy_labels.append(f'validation_{validate_accuracy:.2f}_accuracy')

        if test_accuracy >= 90:
            accuracy_labels.append(f'test_{test_accuracy:.2f}_accuracy')

        # Define the save directory and ensure it exists
        save_dir = '/Users/subhikshaswaminathan/Desktop/Everything/Models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model if any accuracy label conditions are met
        if accuracy_labels:
            model_save_path = os.path.join(save_dir, f'model_{"_".join(accuracy_labels)}.pth')
            torch.save(trained_model, model_save_path)  # Save entire model including architecture
            print(f"Model saved to {model_save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
