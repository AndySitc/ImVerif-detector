import sys
import time
import os
import csv
import torch
from networks.freqnet import freqnet
import numpy as np
import random
import argparse
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path

def random_square_crop(image):
    """Applies a random square crop to a PIL image"""
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

# Custom dataset
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
            print(f"Image loading error: {path}, {e}")
            return None, path

def predict(model, input_path, device, multiple=True, output_file="output/freqnet_test.csv", batch_size=64):
    model.to(device)
    model.eval()
    threshold = 0.5

    # Load existing predictions if available
    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file)
        already_done = set(old_df["image_path"].tolist())
        print(f"{len(already_done)} images already processed.")
    else:
        old_df = pd.DataFrame()
        already_done = set()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "score", "predicted_label"])
            writer.writeheader()

    # Read input paths
    if multiple:
        with open(input_path, "r") as f:
            all_paths = [line.strip() for line in f if line.strip()]
    else:
        all_paths = [input_path]
    
    to_predict = [p for p in all_paths if p not in already_done]

    print(f"🔍 {len(to_predict)} images to predict")

    if not to_predict:
        print("No new images to predict.")
        return

    # Define image transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda img: random_square_crop(img)),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = ImagePathDataset(to_predict, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    fake_list = [
        'fake', 'fake-test-AL', 'DF40', 'DF40_train', 'defacto_copymove', 'defacto_face',
        'defacto_inpainting', 'defacto_splicing', 'cips', 'denoising_diffusion_gan',
        'diffusion_gan', 'face_synthetics', 'gansformer', 'lama', 'mat', 'palette',
        'projected_gan', 'sfhq', 'stable_diffusion', 'star_gan', 'stylegan1',
        'stylegan2', 'stylegan3', 'taming_transformer', 'big_gan'
    ]

    for images, paths in tqdm(dataloader, desc="Batch prediction"):
        valid = [i for i, img in enumerate(images) if img is not None]
        if not valid:
            continue

        images = torch.stack([images[i] for i in valid]).to(device)
        paths = [paths[i] for i in valid]

        with torch.no_grad():
            outputs = model(images)
            outputs = outputs.detach()
            probs = torch.sigmoid(outputs)
            scores = probs.cpu().numpy()
            preds = (scores >= threshold).astype(int)

        # Append predictions to CSV
        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "score", "predicted_label"])
            for path, score, pred in zip(paths, scores, preds):
                parts = Path(path).parts
                correct_label = 1 if any(f in parts for f in fake_list) else 0
                writer.writerow({
                    "image_path": path,
                    "score": float(score),
                    "predicted_label": int(pred),
                    # "correct_label": correct_label
                })

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    print("Entering FreqNet prediction script...")

    parser = argparse.ArgumentParser(description="Prediction script for deepfake image detection using FreqNet.")
    parser.add_argument("--input", required=True, help="Path to a .txt file listing image paths to predict.")
    parser.add_argument("--multiple", action="store_true", help="If set, reads multiple image paths from file.")
    args = parser.parse_args()
    
    model = freqnet(num_classes=1)

    # Load model weights
    checkpoint_path = "models/freqNet/freqnet.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run prediction
    predict(
        model=model,
        input_path=args.input,
        device=device,
        multiple=args.multiple,
        output_file="output/freqnet.csv",
        batch_size=64
    )
