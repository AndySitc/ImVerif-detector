import os
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


from default_config import ModelModes, ModelTypes, hific_args, directories
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes
from src.model import Model  # your full HiFiC model
from src.df_det import HiFiCDeepfakeClassifier
from src.read_data import FullyGenDB
from src.helpers import utils, datasets

from torch.utils.tensorboard import SummaryWriter





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
    general.add_argument("--epoch_restart", type=int, help="Epoch you wish to restart from.")

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


    writer = SummaryWriter(os.path.join('output', 'Compression4DFdet', 'tensorboard_output'))
    device = 'cuda'
    # Initialize HiFiC in EVALUATION mode
    hific = Model(hific_args, logger, model_mode=ModelModes.EVALUATION, model_type=ModelTypes.COMPRESSION_GAN)
    # hific.load_state_dict(torch.load(os.path.join('models', 'hific_low.pt'), weights_only=True))
    hific.eval().to(device)

    # Build classifier
    classifier = HiFiCDeepfakeClassifier(hific_model=hific).to(device)
    if args.epoch_restart is not None:
        classifier.hific.Hyperprior.hyperprior_entropy_model.build_tables()
        classifier.load_state_dict(torch.load("output/Compression4DFdet/epoch_46.pth"))

    # Dataset
    train_data = FullyGenDB(datadir=r'/medias/db/ImagingSecurity_misc/sitcharn/ArtiFactDB', mode='train')
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_data = FullyGenDB(datadir=r'/medias/db/ImagingSecurity_misc/sitcharn/ArtiFactDB', mode='test')
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

    # Criterion
    criterion = nn.BCELoss()
    # Optimizer
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 1. Warmup for 10 epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    # 2. Cosine annealing for 90 epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=90)
    # 3. Combine using SequentialLR
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])


    n_epoch = 101
    first_epoch = 0 if args.epoch_restart is None else args.epoch_restart
    for epoch in range(n_epoch):
        if epoch < first_epoch:
            scheduler.step()
            print(f"Skip epoch {epoch}")
            continue
        total_loss = 0.0
        correct = 0
        total = 0
        predicted = 0

        all_preds = []
        all_labels = []
        for idx, (img, bpp, label) in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch}/{n_epoch}")):
            img = img.to(device)
            label = label.to(device).float().squeeze()  # Ensure correct shape for BCELoss

            preds = classifier(img).squeeze()
            loss = criterion(preds, label)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # batch-Metrics
            batch_loss = loss.item() * img.size(0)
            batch_predicted = (preds > 0.5).float()
            batch_correct = (batch_predicted == label).sum().item()
            batch_total = label.size(0)
            batch_accuracy = batch_correct/batch_total * 100

            # overall-Metrics
            total_loss += batch_loss
            predicted += batch_predicted.sum().item()
            correct += batch_correct
            total += batch_total

            writer.add_scalar("ACCtrain-batch", batch_accuracy, epoch*len(train_dataloader)+idx)
            writer.add_scalar("Losstrain-batch", batch_loss, epoch*len(train_dataloader)+idx)
            try:
                batch_roc_auc = roc_auc_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy())
                writer.add_scalar("AUCtrain-batch", batch_roc_auc, epoch*len(train_dataloader)+idx)
            except:
                print('[TRAIN] Only one class present for this batch. AUC not computed.')  # In case only one class present in this epoch
            # Store predictions and labels for ROC AUC
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())
        scheduler.step()
            
        avg_loss = total_loss / total
        accuracy = correct / total * 100
        
        # Compute ROC AUC
        try:
            roc_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            roc_auc = float('nan')  # In case only one class present in this epoch

        writer.add_scalar("ACCtrain-epoch", accuracy, epoch)
        writer.add_scalar("LOSStrain-epoch", avg_loss, epoch)
        writer.add_scalar("AUCtrain-epoch", roc_auc, epoch)
        logger.info(f"Train Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%, ROC AUC = {roc_auc:.4f}")

        torch.save(classifier.state_dict(), 'output/Compression4DFdet/latest.pth')

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            predicted = 0

            all_preds = []
            all_labels = []
            for idx, (img, bpp, label) in enumerate(tqdm(test_dataloader, desc=f"Test Epoch {epoch}/{n_epoch}")):
                img = img.to(device)
                label = label.to(device).float().squeeze()  # Ensure correct shape for BCELoss

                preds = classifier(img).squeeze()
                loss = criterion(preds, label)

                
                # batch-Metrics
                batch_loss = loss.item() * img.size(0)
                batch_predicted = (preds > 0.5).float()
                batch_correct = (batch_predicted == label).sum().item()
                batch_total = label.size(0)
                batch_accuracy = batch_correct/batch_total * 100

                # overall-Metrics
                total_loss += batch_loss
                predicted += batch_predicted.sum().item()
                correct += batch_correct
                total += batch_total


                writer.add_scalar("ACCtest-batch", batch_accuracy, epoch*len(train_dataloader)+idx)
                writer.add_scalar("LOSStest-batch", batch_loss, epoch*len(train_dataloader)+idx)
                try:
                    batch_roc_auc = roc_auc_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy())
                    writer.add_scalar("AUCtest-batch", batch_roc_auc, epoch*len(train_dataloader)+idx)
                except:
                    print('[TEST] Only one class present for this batch. AUC not computed.')  # In case only one class present in this epoch
                # Store predictions and labels for ROC AUC
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(label.detach().cpu().numpy())
                
            avg_loss = total_loss / total
            accuracy = correct / total * 100
            # Compute ROC AUC
            try:
                roc_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                roc_auc = float('nan')  # In case only one class present in this epoch
            writer.add_scalar("ACCtest-epoch", accuracy, epoch)
            writer.add_scalar("LOSStest-epoch", avg_loss, epoch)
            writer.add_scalar("AUCtest-epoch", roc_auc, epoch)
            logger.info(f"Test Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%, ROC AUC = {roc_auc:.4f}")