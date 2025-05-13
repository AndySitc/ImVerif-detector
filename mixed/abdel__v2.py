import torch

# Device configuration
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("Running on CPU. This will be significantly slower.")


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import csv
import gc
import argparse
from train_deepfake_detector import ColorAwareResNet, EnsembleModel

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, path


def test_batch_images(
    device,
    input_path,
    output_csv,
    multiple,
    batch_size=16,
    model_path="models/mixed/colorbest_deepfake_detector.pth"
):
    # Load model
    print("Loading model...")
    models = [ColorAwareResNet(num_classes=2) for _ in range(3)]
    model = EnsembleModel(models)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model successfully loaded.")

    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # List of known fake categories
    fake_list = ['fake', 'fake-test-AL', 'DF40', 'DF40_train', 'defacto_copymove', 'defacto_face', 'defacto_inpainting', 'defacto_splicing', 'cips', 'denoising_diffusion_gan', 'diffusion_gan',
                 'face_synthetics', 'gansformer', 'lama', 'mat', 'palette', 'projected_gan', 'sfhq',
                 'stable_diffusion', 'star_gan', 'stylegan1', 'stylegan2', 'stylegan3', 'taming_transformer']

    # Read already predicted images
    already_predicted = set()
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        already_predicted.update(df_existing["image_path"].tolist())
        print(f"{len(already_predicted)} images already predicted.")

    if multiple:
        # Read all image paths from input file
        with open(input_path, "r") as f:
            all_paths = [line.strip() for line in f.readlines()]
        remaining_paths = [p for p in all_paths if p not in already_predicted]
        print(f"{len(remaining_paths)} images remaining to predict.")

        # Dataset and DataLoader
        dataset = ImagePathDataset(remaining_paths, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create CSV header if file does not exist
        if not os.path.exists(output_csv):
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "score", "predicted_label"])
                writer.writeheader()

        print("Starting batch predictions...")
        for images, paths in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)

            with torch.no_grad():
                outputs, _ = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                fake_probs = probs[:, 0].cpu().numpy()
                real_probs = probs[:, 1].cpu().numpy()

            with open(output_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "score", "predicted_label"])
                for path, fake_prob, real_prob in zip(paths, fake_probs, real_probs):
                    predicted_label = 0 if real_prob > 0.5 else 1
                    correct_label = 1 if any(f in Path(path).parts for f in fake_list) else 0
                    writer.writerow({
                        "image_path": path,
                        "score": float(fake_prob),
                        "predicted_label": predicted_label,
                        # "correct_label": correct_label
                    })

            del images, outputs, probs
            torch.cuda.empty_cache()
            # gc.collect()

        print("All predictions are complete.")
    else:
        image = Image.open(input_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        results = []

        # Single image prediction
        with torch.no_grad():
            outputs, _ = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get probabilities
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()

            # Get prediction
            prediction = 0 if real_prob > 0.5 else 1
            confidence = max(real_prob, fake_prob) * 100

            parts = Path(input_path).parts

            results.append({
                "image_path": input_path,
                "score": float(fake_prob),
                "predicted_label": prediction,
                # "correct_label": 1 if any(f in parts for f in fake_list) else 0
            })

            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Deepfake image prediction script.")
    parser.add_argument("--input", required=True, help="Text file containing paths of images to predict")
    parser.add_argument("--multiple", action="store_true", help="If enabled, processes multiple images listed in the file")
    args = parser.parse_args()

    test_batch_images(
        device,
        input_path=args.input,
        multiple=args.multiple,
        output_csv="output/abdel.csv",
        batch_size=16
    )
