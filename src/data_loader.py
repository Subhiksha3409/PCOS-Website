import os
from torch.utils.data import Dataset
from torchvision import transforms

class PCOSDataset(Dataset):
    """
    Custom Dataset class for PCOS image classification.
    """
    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if not self.image_files:
            raise ValueError(f"No images found in the folder {image_folder}. Make sure the folder contains images and the path is correct.")

        print(f"Found {len(self.image_files)} images in {image_folder}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        print(f"Loading image from {img_name}")  # Debug: Print image path
        
        try:
            image = Image.open(img_name).convert('RGB')
        except UnidentifiedImageError:
            print(f"Warning: Cannot identify image file {img_name}. Skipping this file.")
            # Optionally, you could return a dummy image and label
            return transforms.ToTensor()(Image.new('RGB', (224, 224))), 0

        if self.transform:
            image = self.transform(image)

        label = 1 if 'PCOS' in self.image_folder else 0

        return image, label

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
