import os
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
from torchvision import transforms, datasets
import pandas as pd


from default_config_test import ModelModes, ModelTypes, hific_args, directories
from default_config_test import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes
from src.model import Model  # your full HiFiC model
from src.df_det import HiFiCDeepfakeClassifier
from src.read_data import FullyGenDB
from src.helpers import utils, datasets
from pathlib import Path
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from tqdm import tqdm
import gc

def apply_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def read_data(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    return apply_transform()(img)

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = list(image_paths)  # 🔧 Conversion en liste pour l'indexation
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        if self.transform:
            image = self.transform(image)
        return image, image_path

def predict(model, input_path, device, multiple, output_file="alex_github.csv", name="Alex"):
    model.to(device)
    model.eval()

    # Lire les prédictions déjà faites
    already_done = set()
    output_path = Path("output") / output_file
    if output_path.exists():
        df_old = pd.read_csv(output_path)
        already_done = set(df_old["image_path"].tolist())
        print(f"📄 {len(already_done)} images déjà traitées")

    fake_list = [
        'fake', 'DF40', 'DF40_train', 'defacto_copymove', 'defacto_face', 'defacto_inpainting', 'defacto_splicing', 'cips',
        'denoising_diffusion_gan', 'diffusion_gan', 'face_synthetics', 'gansformer', 'lama', 'mat', 'palette',
        'projected_gan', 'sfhq', 'stable_diffusion', 'star_gan', 'stylegan1', 'stylegan2', 'stylegan3', 'taming_transformer'
    ]

    def append_to_csv(row):
        header = not output_path.exists()
        with open(output_path, "a") as f:
            pd.DataFrame([row]).to_csv(f, header=header, index=False)

    print("🔍 Démarrage des prédictions...")

    if multiple:
        print("Mode batch activé...")

        # 🔍 Récupération de toutes les images dans le dossier input_path
        with open(input_path, "r") as f:
            all_image_paths = [line.strip() for line in f if line.strip()]
        new_images = list(set(all_image_paths) - already_done)

        dataset = ImageDataset(image_paths=new_images, transform=apply_transform())
        if len(dataset) == 0:
            print("✅ Aucune nouvelle image à prédire.")
            return

        dataloader = DataLoader(dataset, shuffle=False, batch_size=8, num_workers=16)

        for images, paths in tqdm(dataloader, desc="📸 Prédictions"):
            images = images.to(device)
            preds = model(images).squeeze()
            scores = preds.detach().cpu().numpy()

            for path, score in zip(paths, scores):
                parts = Path(path).parts
                predicted_label = int(score > 0.5)
                correct_label = int(any(f in parts for f in fake_list))
                row = {
                    "image_path": path,
                    "score": float(score),
                    "predicted_label": predicted_label,
                    # "correct_label": correct_label
                }
                append_to_csv(row)

            # nettoyage mémoire
            del images, preds
            torch.cuda.empty_cache()
            gc.collect()
        print(f"✅ Prédictions terminées. Résultats dans {output_path}")

    else:
        if input_path in already_done:
            print(f"⏭️ Image déjà prédite : {input_path}")
            return

        image = read_data(image_path=input_path).to(device).unsqueeze(0)
        score = model(image).squeeze().item()
        predicted_label = int(score > 0.5)
        parts = Path(input_path).parts
        correct_label = int(any(f in parts for f in fake_list))
        row = {
            "image_path": input_path,
            "score": float(score),
            "predicted_label": predicted_label,
            # "correct_label": correct_label
        }
        append_to_csv(row)
        print(f"✅ Prédiction enregistrée dans {output_path}")

    

if __name__ == '__main__':
    description = "CompressionForDeepfake-Detection"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-mt", "--model_type", required=True, choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN), 
        help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low','med','high'), default='low', help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=hific_args.log_interval, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=hific_args.save_interval, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument("-norm", "--normalize_input_image", help="Normalize input images to [-1,1]", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=hific_args.batch_size, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument("-lt", "--likelihood_type", choices=('gaussian', 'logistic'), default='gaussian', help="Likelihood model for latents.")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")
    general.add_argument("-LMM", "--use_latent_mixture_model", help="Use latent mixture model as latent entropy model.", action="store_true")
    general.add_argument("-input", "--input", type=str, default=None, help="The image path")
    general.add_argument("--multiple", action="store_true", default=False, help="Use this flag to enable batch prediction")


    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=float, default=hific_args.n_steps, 
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=hific_args.n_epochs, 
        help="Number of passes over training dataset. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=hific_args.learning_rate, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=hific_args.weight_decay, help="Coefficient of L2 regularization.")

    # Architecture-related options
    arch_args = parser.add_argument_group("Architecture-related options")
    arch_args.add_argument('-lc', '--latent_channels', type=int, default=hific_args.latent_channels,
        help="Latent channels of bottleneck nominally compressible representation.")
    arch_args.add_argument('-nrb', '--n_residual_blocks', type=int, default=hific_args.n_residual_blocks,
        help="Number of residual blocks to use in Generator.")

    # Warmstart adversarial training from autoencoder/hyperprior
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart adversarial training from autoencoder + hyperprior ckpt.", action="store_true")
    warmstart_args.add_argument("-ckpt", "--warmstart_ckpt", default=None, help="Path to autoencoder + hyperprior ckpt.")

    cmd_args = parser.parse_args()
    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))


    device = f'cuda:{cmd_args.gpu}'
    print(f"🚀 Using device: {device}")

    # Initialize HiFiC in EVALUATION mode
    hific = Model(hific_args, logger, model_mode=ModelModes.EVALUATION, model_type=ModelTypes.COMPRESSION_GAN)
    # hific.load_state_dict(torch.load(os.path.join('models', 'hific_low.pt'), weights_only=True))
    hific.eval().to(device)

    # Build classifier
    classifier = HiFiCDeepfakeClassifier(hific_model=hific).to(device)

    classifier.hific.Hyperprior.hyperprior_entropy_model.build_tables()
    classifier.load_state_dict(torch.load("models/fully_generated/latest.pth"))
    classifier.eval()
    predict(classifier, cmd_args.input, device, cmd_args.multiple, output_file="alex_test_dataset_balanced.csv")