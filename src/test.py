import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import PCOSDataset, transform
from src.PCOS_Website.model import SimpleCNN

def test_model(test_loader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    return accuracy
