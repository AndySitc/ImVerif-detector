import os
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
from torchvision import transforms, datasets
import pandas as pd


from default_config import ModelModes, ModelTypes, hific_args, directories
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes
from src.model import Model  # your full HiFiC model
from src.df_det import HiFiCDeepfakeClassifier
from src.read_data import FullyGenDB
from src.helpers import utils, datasets

def apply_transform():
    transforms_list = [transforms.ToTensor()]
    transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transforms_list)

def read_data(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        transformed = apply_transform()(img)
        return transformed

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
    general.add_argument("-image_path", "--image_path", type=str, default=None, help="The image path")

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
    # Initialize HiFiC in EVALUATION mode
    hific = Model(hific_args, logger, model_mode=ModelModes.EVALUATION, model_type=ModelTypes.COMPRESSION_GAN)
    # hific.load_state_dict(torch.load(os.path.join('models', 'hific_low.pt'), weights_only=True))
    hific.eval().to(device)

    # Build classifier
    classifier = HiFiCDeepfakeClassifier(hific_model=hific).to(device)


    # Path of test image
    # img_path = '/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg'
    img = read_data(image_path=cmd_args.image_path)
    classifier.hific.Hyperprior.hyperprior_entropy_model.build_tables()
    classifier.load_state_dict(torch.load("HiFic/high-fidelity-generative-compression-master/high-fidelity-generative-compression-master/output/Compression4DFdet/latest.pth"))
    classifier.eval()

    # Inference
    img = img.to(device).unsqueeze(0)
    preds = classifier(img).squeeze()
    print(f"Score is {preds.detach().cpu().numpy()}. Prediction: {'Fake' if preds.detach().cpu().numpy() > 0.5 else 'Real'}")

    results = []
    results.append({
        "image_path": cmd_args.image_path,
        "probability": preds.detach().cpu().numpy(),
        "binary_result": 1 if preds.detach().cpu().numpy() > 0.5 else 0,
        "threshold": 0.5,
        "detector": "Alex",
    })

    output_file = "alex.csv"
    # Créer un DataFrame
    df = pd.DataFrame(results)
    df.to_csv("output/"+output_file, index=False)
    print(f"Prédictions sauvegardées dans output/{output_file}")