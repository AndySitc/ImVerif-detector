import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from torch.utils.data import Dataset
from PIL import Image
import pickle
from tqdm import tqdm
from code_model import VITContrastiveHF
from dataset_paths_imverif import DATASET_PATHS_IMVERIF
import random
from pathlib import Path
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset





# Custom dataset
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path






@torch.inference_mode()
def predict(device, input_path, multiple, model, output_csv, batch_size=16):
    # Transformations d'image
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Liste des images à prédire
    if multiple:
        with open(input_path, "r") as f:
            all_paths = [line.strip() for line in f if line.strip()]
    else:
        all_paths = [input_path]

    # Dataset et DataLoader
    dataset = ImagePathDataset(all_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Préparer le fichier de sortie
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path", "score", "predicted_label"])

        # Prédiction par batch
        model.eval()
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Batch prediction"):
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                fake_probs = probs[:, 1].cpu().numpy()
                preds = (fake_probs > 0.5).astype(int)

                for path, prob, pred in zip(paths, fake_probs, preds):
                    writer.writerow([path, prob, pred])
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction script for deepfake images.")
    parser.add_argument("--input", required=True, help="Path to image or .txt file containing image paths")
    parser.add_argument("--multiple", action="store_true", help="If set, treats the input as a list of images")
    args = parser.parse_args()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = VITContrastiveHF(classificator_type="linear")
    model.eval()
    model.to(device)

    # Run prediction
    predict(
        device,
        input_path=args.input,
        multiple=args.multiple,
        model=model,
        output_csv="output/CoDE.csv"
    )
