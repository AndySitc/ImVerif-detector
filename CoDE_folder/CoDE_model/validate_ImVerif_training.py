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


def set_seed(SEED=0):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def calculate_acc_svm(y_true, y_pred):
    y_true[y_true == 1] = -1
    y_true[y_true == 0] = 1
    r_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])  # > thres)
    f_acc = accuracy_score(y_true[y_true == -1], y_pred[y_true == -1])  # > thres)
    acc = accuracy_score(y_true, y_pred)  # > thres)
    return r_acc, f_acc, acc


def write_csv(image_paths, scores, labels, predictions, csv_path="CoDE_test_result.csv"):
    """
    Appends inference results to a CSV file.

    Args:
        image_paths (list of str): Paths to input images.
        scores (list of float): Model confidence scores.
        labels (list of int): Ground-truth labels.
        predictions (list of int): Predicted labels (e.g., 0 or 1).
        csv_path (str): Output CSV file path.
    """
    # Check if file exists to write headers only once
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow(["image_path", "score", "label", "prediction"])

        # Write each row of data
        for path, score, label, pred in zip(image_paths, scores, labels, predictions):
            writer.writerow([path, score, label, pred])

@torch.inference_mode()
def validate(model, loader, opt=None):
    with torch.no_grad():
        print("Length of dataset: %d" % (len(loader)))
        for img, label, img_paths in tqdm(loader):
            in_tens = img.cuda()
            write_csv(img_paths, model(in_tens).flatten().tolist(), label.flatten().tolist(), (model(in_tens).flatten() > 0.5).to(int).flatten().tolist())
    return
    # return image_path_list, score_list, label_list, label_predicted_list


# = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = #


def recursively_read(
    rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "JPG", "tiff"]
):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=""):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

def label_from_filename(path):
    # This function determines the label based on the filename based on ArtiFactDB dataset.
    fake_list = ['fake', 'DF40', 'DF40_train', 'defacto_copymove', 'defacto_face', 'defacto_inpainting', 'defacto_splicing', 'cips', 'denoising_diffusion_gan', 'diffusion_gan',
             'face_synthetics', 'gansformer', 'lama', 'mat', 'palette', 'projected_gan', 'sfhq',
             'stable_diffusion', 'star_gan', 'stylegan1', 'stylegan2', 'stylegan3', 'taming_transformer']
    parts = Path(path).parts  # tuple des répertoires
    label = int(any(f in parts for f in fake_list))
    return label

class RealFakeDataset(Dataset):
    def __init__(self, txt_file, transform, label_provider=None):
        """
        Args:
            txt_file (str): Path to the .txt file with image paths.
            transform (callable): Transform to apply to images.
            label_provider (callable or dict, optional): Function or dict to determine label from image path.
        """
        # Read all image paths from the txt file
        with open(txt_file, 'r') as f:
            self.total_list = [line.strip() for line in f if line.strip()]
        
        # Prepare labels based on a provided label function or dict
        self.labels_dict = {}
        if label_provider:
            for path in self.total_list:
                self.labels_dict[path] = label_provider(path)
        else:
            # Default to label 0 (you can modify this logic as needed)
            self.labels_dict = {path: 0 for path in self.total_list}

        self.transform = transform

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label, img_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--real_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument(
        "--fake_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument("--data_mode", type=str, default=None, help="wang2020 or ours")
    parser.add_argument("--result_folder", type=str, default="./results", help="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--classificator_type", type=str, default="linear")
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)

    model = VITContrastiveHF(classificator_type=opt.classificator_type)

    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("Model loaded..")
    model.eval()
    model.cuda()
    dataset_paths = DATASET_PATHS_IMVERIF
    for dataset_path in dataset_paths:
        set_seed()
        print(f"dataset_path: {dataset_path}")
        dataset = RealFakeDataset(
            r"/medias/db/ImagingSecurity_misc/Collaborations/ImVerif_Detector 2/data/liste_images_test.txt",
            transform=transform,
            label_provider=label_from_filename
            
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
        )
        r_acc0, f_acc0, acc0, auc = validate(model, loader, opt=opt)

        
