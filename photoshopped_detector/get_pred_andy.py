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
from PIL import Image
import torch
import numpy as np
import pandas as pd
import os
# import settings  # Assure-toi que settings contient `DEVICE`
# from transformers import AutoModel, AutoTokenizer, util



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

def predict_and_save(model, image_path, device, output_file="andy.csv"):
    
    model.to(device)
    model.eval()  
    results = []

    # all_labels = []  
    all_predictions = []  
    
    # Définir le vrai label à partir du chemin (FAKE = 1, REAL = 0)
    # true_label = 1 if "FAKE" in image_path else 0  
    # all_labels.append(true_label)
    # print(image_path)
    image = load_image(image_path, transforms.ToTensor()).unsqueeze(0).to(device)
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    score = probs[:, 1]  

    threshold = 0.9889
    predicted_label = int(score >= threshold)
    all_predictions.append(predicted_label)

    results.append({
        "image_path": image_path,
        "probability": float(score),
        "binary_result": predicted_label,
        "threshold": threshold,
        "detector": "Andy",
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
    # Gestion des arguments CLI
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--output", type=str, required=True, help="Chemin du fichier de sortie")
    # args = parser.parse_args()


    # Choix automatique du GPU
    # available_gpus = get_available_gpu()
    # if len(available_gpus) > 0:
    #     chosen_gpu = available_gpus[0]
    # else:
    #     raise RuntimeError("Aucun GPU disponible avec suffisamment de mémoire.")

    # # Fixe le GPU visible pour tout le script
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)

    # # Optionnel : Set le device explicitement
    # torch.cuda.set_device(0)  # car CUDA_VISIBLE_DEVICES=1 => 0 est mappé à l'ancien GPU 1

    # Vérifier si un argument a été fourni
    if len(sys.argv) < 2:
        print("Usage: python predict_env1.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]  # Récupérer l'image passée en argument
    print(f"Prédiction pour l'image : {image_path}")

    # Charger l'image
    # image = Image.open(image_path)


    # 🔹 Recréer l'architecture du modèle
    model = EfficientNetClassifier(num_classes=2)  # ⚠️ Doit être exactement comme celui utilisé à l'entraînement

    # 🔹 Charger les poids entraînés
    model.load_state_dict(torch.load(
        "/medias/ImagingSecurity_misc/sitcharn2/outputs/ckpt/2025-04-08_14-01-17/epoch_9.pth",
        map_location=torch.device("cpu")
    ))

    # 🔹 Mettre en mode évaluation
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_and_save(model, image_path, device)
    # print("-" * 50)



