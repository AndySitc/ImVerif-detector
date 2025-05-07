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
import torch
import numpy as np
import pandas as pd
import os



class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetClassifier, self).__init__()

        # Charger EfficientNet pré-entraîné et enlever la couche FC
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b0")
        
        # Remplacer la première couche conv si nécessaire pour des images 200x200
        self.efficientnet._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Classifieur CNN (3 couches de convolution + Fully Connected)
        self.conv1 = nn.Conv2d(1280, 128, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)  # Ajout de dropout
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.3)  # Ajout de dropout
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.ReLU()

        # Ajuster la taille de la couche linéaire après le pooling
        # La sortie est de taille [batch_size, 128, 1, 1] après les convolutions et le pooling
        self.fc1 = nn.Linear(1152, num_classes)  # On a 128 caractéristiques à entrer dans fc1 4608
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Extraction des caractéristiques avec EfficientNet (sans FC)
        x = self.efficientnet.extract_features(x)  # Extraction des features sans passer par la couche FC
        
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # print(f"Shape before flatten: {x.shape}")  # 🔍 Ajoute ceci pour voir la taille du tenseur
        
        x = torch.flatten(x, 1)  # Aplatir avant fully connected
        # print(f"Shape after flatten: {x.shape}")  # 🔍 Vérifier la nouvelle taille après flatten

        x = self.fc1(x)  # ⚠️ Erreur possible ici si les dimensions ne matchent pas
        return x

def random_square_crop(image):
        """Applique un crop carré aléatoire à une image PIL"""
        width, height = image.size

        crop_size = min(width, height)  # Prend la plus petite dimension
        left = torch.randint(0, width - crop_size + 1, (1,)).item()
        top = torch.randint(0, height - crop_size + 1, (1,)).item()
        return TF.crop(image, top, left, crop_size, crop_size)  # Crop carré

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = random_square_crop(image)
    image = TF.resize(image, (200, 200))
    if transform:
        image = transform(image)
    return image

def predict(model, input_path, device, multiple, output_file="andy.csv", name="Andy"):
    
    model.to(device)
    model.eval()  
    results = []

    all_predictions = []  
    fake_list = ['fake', 'DF40', 'DF40_train', 'defacto_copymove', 'defacto_face', 'defacto_inpainting', 'defacto_splicing', 'cips', 'denoising_diffusion_gan', 'diffusion_gan',
             'face_synthetics', 'gansformer', 'lama', 'mat', 'palette', 'projected_gan', 'sfhq',
             'stable_diffusion', 'star_gan', 'stylegan1', 'stylegan2', 'stylegan3', 'taming_transformer']

    
    print("Start of the prediction...")
    if multiple:
        print("multiple prediction...")
        with open(input_path, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
        for image_path in image_paths:
            # i=i+1
            # print(i)
            # print(image_path)
            image = load_image(image_path, transforms.ToTensor()).unsqueeze(0).to(device)
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            score = probs[:, 1]  

            threshold = 0.9889
            # threshold = 0.6739
            predicted_label = int(score >= threshold)
            all_predictions.append(predicted_label)
            parts = Path(image_path).parts  # tuple des répertoires

            results.append({
                "image_path": image_path,
                "score": float(score),
                "predicted_label": predicted_label,
                "correct_label": 1 if any(f in parts for f in fake_list) else 0
            })

            df = pd.DataFrame(results)
            df.to_csv("output/"+output_file, index=False)
            print(f"Prédictions sauvegardées dans output/{output_file}")
        
    else:
        image = load_image(input_path, transforms.ToTensor()).unsqueeze(0).to(device)
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        score = probs[:, 1]  

        threshold = 0.9889
        predicted_label = int(score >= threshold)
        all_predictions.append(predicted_label)
        parts = Path(image_path).parts  # tuple des répertoires

        results.append({
                "image_path": image_path,
                "score": float(score),
                "predicted_label": predicted_label,
                "correct_label": 1 if any(f in parts for f in fake_list) else 0
            })
        print(f"The image is {predicted_label} with a probability of: {float(score)}")

        # Créer un DataFrame
        df = pd.DataFrame(results)
        df.to_csv("output/"+output_file, index=False)
        print(f"Prédictions sauvegardées dans output/{output_file}")
    

# Vérifie la mémoire dispo sur chaque GPU
def get_available_gpu(threshold_ratio=0.95):
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.memory_reserved(i) / torch.cuda.get_device_properties(i).total_memory
        if stats < threshold_ratio:
            free_gpus.append(i)
    return free_gpus


if __name__ == "__main__":
    print("enter in andy script")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--multiple", action="store_true")
    args = parser.parse_args()

    # 🔹 Recréer l'architecture du modèle
    model = EfficientNetClassifier(num_classes=2)  # ⚠️ Doit être exactement comme celui utilisé à l'entraînement
    # model.load_state_dict(torch.load("/medias/db/ImagingSecurity_misc/sitcharn2/outputs/ckpt/2025-04-08_14-01-17/epoch_0.pth",
    #     map_location=torch.device("cpu")))
    model.load_state_dict(torch.load("/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/photoshop_detection/outputs/ckpt/2025-02-07_15-44-07/epoch_3.pth",
        map_location=torch.device("cpu")))
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict(model, args.input, device, args.multiple)