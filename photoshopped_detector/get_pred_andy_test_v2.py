import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import numpy as np
import os
import csv


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetClassifier, self).__init__()

        # Load pretrained EfficientNet and remove the FC layer
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b0")
        
        # Replace the initial conv layer if needed for 200x200 images
        self.efficientnet._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # CNN classifier (3 conv layers + Fully Connected)
        self.conv1 = nn.Conv2d(1280, 128, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.ReLU()

        # Adjust the size of the linear layer after pooling
        self.fc1 = nn.Linear(1152, num_classes)

    def forward(self, x):
        # Feature extraction with EfficientNet (without FC)
        x = self.efficientnet.extract_features(x)
        
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def random_square_crop(image):
    """Apply a random square crop to a PIL image"""
    width, height = image.size
    crop_size = min(width, height)
    left = torch.randint(0, width - crop_size + 1, (1,)).item()
    top = torch.randint(0, height - crop_size + 1, (1,)).item()
    return TF.crop(image, top, left, crop_size, crop_size)


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = random_square_crop(image)
    image = TF.resize(image, (200, 200))
    if transform:
        image = transform(image)
    return image


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, path
        except Exception as e:
            print(f"Error loading image: {path}, {e}")
            return None, path


def predict(model, input_path, device, multiple=True, output_file="output/andy.csv", name="Andy", batch_size=64):
    model.to(device)
    model.eval()

    # Load existing predictions if available
    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file)
        already_done = set(old_df["image_path"].tolist())
        print(f"{len(already_done)} images already processed")
    else:
        old_df = pd.DataFrame()
        already_done = set()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "score", "predicted_label"])
            writer.writeheader()

    # Read paths from input file
    if multiple:
        with open(input_path, "r") as f:
            all_paths = [line.strip() for line in f if line.strip()]
    else:
        all_paths = [input_path]

    to_predict = [p for p in all_paths if p not in already_done]
    print(f"{len(to_predict)} images to predict")

    if not to_predict:
        print("No new images to predict.")
        return

    # Transformations compatible with the model (200x200 + normalization)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: random_square_crop(img)),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = ImagePathDataset(to_predict, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    fake_list = ['fake', 'fake-test-AL', 'DF40', 'DF40_train', 'defacto_copymove', 'defacto_face', 'defacto_inpainting',
                 'defacto_splicing', 'cips', 'denoising_diffusion_gan', 'diffusion_gan', 'face_synthetics',
                 'gansformer', 'lama', 'mat', 'palette', 'projected_gan', 'sfhq', 'stable_diffusion',
                 'star_gan', 'stylegan1', 'stylegan2', 'stylegan3', 'taming_transformer', 'big_gan']

    threshold = 0.9889

    for images, paths in tqdm(dataloader, desc="Batch prediction"):
        valid = [i for i, img in enumerate(images) if img is not None]
        if not valid:
            continue

        images = torch.stack([images[i] for i in valid]).to(device)
        paths = [paths[i] for i in valid]

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            scores = probs[:, 1].cpu().numpy()
            preds = (scores >= threshold).astype(int)

        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "score", "predicted_label"])
            for path, score, pred in zip(paths, scores, preds):
                parts = Path(path).parts
                correct_label = 1 if any(f in parts for f in fake_list) else 0
                writer.writerow({
                    "image_path": path,
                    "score": float(score),
                    "predicted_label": int(pred),
                })

    print(f"Results saved to {output_file}")


def get_available_gpu(threshold_ratio=0.95):
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.memory_reserved(i) / torch.cuda.get_device_properties(i).total_memory
        if stats < threshold_ratio:
            free_gpus.append(i)
    return free_gpus


if __name__ == "__main__":
    print("Starting Andy's prediction script")

    parser = argparse.ArgumentParser(description="Deepfake image prediction script.")
    parser.add_argument("--input", required=True, help="Text file with image paths to predict")
    parser.add_argument("--multiple", action="store_true", help="If enabled, processes multiple images from file")
    args = parser.parse_args()

    # Load the model
    model = EfficientNetClassifier(num_classes=2)

    # Load model weights
    checkpoint_path = "models/photoshopped/epoch_3.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Launch prediction
    predict(
        model=model,
        input_path=args.input,
        device=device,
        multiple=args.multiple,
        output_file="output/andy.csv",
        batch_size=64
    )
