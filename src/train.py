import os
import torch
import torch.optim as optim
import torch.nn as nn
from src.PCOS_Website.model import SimpleCNN
from torch.utils.data import DataLoader

def train_model(train_loader):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_correct = 0
    total_samples = 0

    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = 100 * total_correct / total_samples
        print(f"Epoch {epoch+1}/10, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # Prepare the save directory
    save_dir = r'C:\Users\raees\Desktop\Models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save model if accuracy is above 90%
    if accuracy >= 90:
        model_save_path = os.path.join(save_dir, f'train_{accuracy:.2f}_accuracy.pth')
        torch.save(model, model_save_path)  # Save entire model including architecture
        print(f"Model saved to {model_save_path}")

    return accuracy, model
