'''
This scripts is for inference; to run the code run:
python evaluation.py --detector_path '/training/config/detector/xception.yaml' 
 --weights_path '/training/pretrained/train_on_df40/xception.pth'
--image_dir '/training/pretrained/train_on_df40/xception.pth'         
'''
import os
import yaml
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from detectors import DETECTOR
# from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import numpy as np
import sys
import time
import cv2
import dlib
import logging
import datetime
import glob
import concurrent.futures
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import argparse
from logger import create_logger

root = str(Path(__file__).resolve().parent)
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default=f'{root}/config/detector/xception.yaml',
                    help='path to detector YAML file')
parser.add_argument('--weights_path', type=str, 
                    default=f'{root}/pretrained/train_on_df40/xception.pth')
parser.add_argument('--input', type=str, 
                    default=f'{root}/DATA')

parser.add_argument('--face_detector_path', type=str, 
                    default='')

parser.add_argument("--multiple", action="store_true", help="Si activé, traite plusieurs images listées dans le fichier")



#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()
# #=============================================
# #adding root to tha paths
# args.detector_path = root+args.detector_path
# args.weights_path= root+args.weights_path
# args.image_dir = root+args.image_dir
# #=============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("Running on CPU. This will be significantly slower.")

#Input preprocessing

face_detector = dlib.get_frontal_face_detector()
predictor_path = args.face_detector_path
print("predictor_path: ", predictor_path)
face_predictor = dlib.shape_predictor(predictor_path)

def compute_and_save_confusion_matrix(results, output_path='confusion_matrix.png'):
    """
    Compute the confusion matrix and classification report, and save the confusion matrix plot.

    Args:
        results (dict): Dictionary containing ground truth labels and predicted probabilities.
                        Format: {'fake': [(image_name, prob), ...], 'real': [(image_name, prob), ...]}
        output_path (str): Path to save the confusion matrix plot.

    Returns:
        None
    """
    # Initialize ground truth and predictions
    ground_truth = []
    predictions = []

    # Iterate through results to collect ground truth and predicted labels
    for label, preds in results.items():
        for image_name, prob in preds:
            # Ground truth: 1 for fake, 0 for real
            ground_truth.append(1 if label == 'fake' else 0)
            # Predicted label: 1 if prob >= 0.5, else 0
            predictions.append(1 if prob >= 0.5 else 0)

    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=['real', 'fake']))

    # Plot confusion matrix with larger font size
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'fake'], yticklabels=['real', 'fake'], annot_kws={"size": 14})
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title('Confusion Matrix', fontsize=25)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Confusion matrix plot saved to: {output_path}")
    plt.close()

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """
        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # Calculate the center of the image
        img_center = np.array([rgb.shape[1] // 2, rgb.shape[0] // 2])

        # For now only take the biggest face and the face closest to the center
        face = max(faces, 
                   key=lambda rect: rect.width() * rect.height() and np.linalg.norm(np.array([(rect.left() + rect.right()) // 2, 
                   (rect.top() + rect.bottom()) // 2]) - img_center)
        )

        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)
        return cropped_face, landmark, mask_face
    else:
        return None, None, None


def load_image(image_path, config, face_detector, face_predictor, resolution=256, save_dir=None):
    """
    Load and preprocess an image, including face alignment and transformations.

    Args:
        image_path (str): Path to the input image.
        config (dict): Configuration dictionary containing preprocessing parameters.
        face_detector: Dlib face detector.
        face_predictor: Dlib shape predictor for facial landmarks.
        resolution (int): Resolution to resize the aligned face.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for inference.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Align and crop the face using dlib
    cropped_face, landmarks, _ = extract_aligned_face_dlib(
        face_detector, face_predictor, image, res=resolution, mask=None
    )
    if cropped_face is None:
        # Convert the cropped face to RGB format
        # cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

        # Apply transformations: resize, convert to tensor, and normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.Resize((config['resolution'], config['resolution'])),  # Resize to target resolution
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=config['mean'], std=config['std'])  # Normalize using mean and std
        ])
        image_tensor = transform(image).unsqueeze(0).to(device) 
        return image_tensor
        # raise ValueError(f"No face detected in the image: {image_path}")

    # Save the cropped face for a sanity check
    # if save_dir:
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, os.path.basename(image_path))
    #     cv2.imwrite(save_path, cropped_face)
    #     print(f"Cropped face saved to: {save_path}")

    # Convert the cropped face to RGB format
    else:
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

        # Apply transformations: resize, convert to tensor, and normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.Resize((config['resolution'], config['resolution'])),  # Resize to target resolution
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=config['mean'], std=config['std'])  # Normalize using mean and std
        ])
        image_tensor = transform(cropped_face).unsqueeze(0).to(device)  # Add batch dimension and move to device
        return image_tensor


@torch.no_grad()
def inference(model, image_tensor):
    """Perform inference on a single image."""
    data_dict = {'image': image_tensor}
    predictions = model(data_dict, inference=True)
    return predictions['prob'].item()


def main():
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # Initialize the model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    init_seed(config)
    epoch = 0
    if args.weights_path:
        try:
            epoch = int(args.weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(args.weights_path, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')  
            if 'base_model.' in new_key:
                new_key = new_key.replace('base_model.', 'backbone.')
            if 'classifier.' in new_key:
                new_key = new_key.replace('classifier.', 'head.')
            new_weights[new_key] = value

        model.load_state_dict(new_weights, strict=True)
        model.eval()
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    # Read image paths from the text file
    text_file_path = args.input  # Use the `--image_dir` argument to specify the text file

    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Text file not found: {text_file_path}")
    
    if args.multiple:
        with open(args.input, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        image_paths = [args.input]

    # with open(text_file_path, 'r') as f:
    #     image_paths = [line.strip() for line in f.readlines()]

    # Initialize results list
    results = []

    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Define a mapping of keywords to labels
            keyword_to_label = {
                '/DF40/': 'fake',
                '/DF40_train/': 'fake',
                '/Celeb-DF-v2/' : 'real',
                'original_sequences': 'real'
            }

            # Determine the correct label based on keywords in the image path
            correct_label = None
            for keyword, mapped_label in keyword_to_label.items():
                if keyword in image_path:
                    correct_label = mapped_label
                    break

            # Default label if no keyword matches
            if correct_label is None:
                correct_label = 'unknown'

            # Load and preprocess the image
            image_tensor = load_image(image_path, config, face_detector, face_predictor, resolution=224)

            # Perform inference
            prob = inference(model, image_tensor)

            # Determine the predicted label based on the probability
            predicted_label = 1 if prob >= 0.5 else 0

            # Append the result to the list
            results.append({
                'image_path': image_path,
                'score': prob,
                'predicted_label': predicted_label,
                # 'correct_label': correct_label
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")    

    # Save CSV
    results_dir = "output/"
    os.makedirs(results_dir, exist_ok=True)

    detector_name = Path(args.weights_path).parts[-2] if args.weights_path else "no_weights"
    csv_file_path = os.path.join(results_dir, f"sahar_{detector_name}.csv")

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image_path", "score", "predicted_label"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Results saved to: {csv_file_path}")

if __name__ == "__main__":
    main()