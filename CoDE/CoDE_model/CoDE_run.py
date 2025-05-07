import argparse
import os
import sys
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from PIL import Image
import pickle
from tqdm import tqdm
from code_model import VITContrastiveHF
from dataset_paths_imverif import DATASET_PATHS_IMVERIF
from sklearn.metrics import roc_curve, roc_auc_score
import random
from pathlib import Path
import csv
import argparse



# @torch.inference_mode()
import torch.nn.functional as F

@torch.inference_mode()
def predict(image_path, model, output_csv):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).cuda()

    # Prédire les logits
    logits = model(img_tensor)

    # Appliquer softmax pour avoir des probabilités
    probs = F.softmax(logits, dim=1)

    # Probabilité que ce soit "fake" (supposée classe 1)
    fake_prob = probs[0][1].item()

    predicted_label = int(fake_prob > 0.5)
    # label = label_from_filename(image_path)

    # Écriture dans le CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path", "score", "predicted_label"])
        writer.writerow([image_path, fake_prob, predicted_label])

    print(f"✅ Image: {image_path} | Score: {fake_prob:.4f} | Prediction: {predicted_label}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de prédiction d'images deepfake.")
    parser.add_argument("--input", required=True, help="Fichier .txt contenant les chemins des images à prédire")
    parser.add_argument("--multiple", action="store_true", help="Si activé, traite plusieurs images listées dans le fichier")
    args = parser.parse_args()

    model = VITContrastiveHF(classificator_type="linear")
    model.eval()
    model.cuda()

    predict(
        image_path=args.input,
        model=model,
        output_csv="output/CoDE_github.csv"
    )
