import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
# class HiFiCDeepfakeClassifier(nn.Module):
#     def __init__(self, hific_model, latent_shape=(320, 32, 32), use_entropy_features=True):
#         """
#         Args:
#             hific_model (Model): The HiFiC model instance (already initialized).
#             latent_shape (tuple): Shape of encoder output 'y', typically (C, H, W).
#             use_entropy_features (bool): Whether to concatenate BPP stats.
#         """
#         super(HiFiCDeepfakeClassifier, self).__init__()
#         self.hific = hific_model
#         self.use_entropy_features = use_entropy_features
#         C, H, W = latent_shape
#         self.latent_feat_dim = C * H * W
#         extra_features = 3 if use_entropy_features else 0
#         input_dim = self.latent_feat_dim + extra_features

#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.3),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Dropout(0.3),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         """
#         Args:
#             x (Tensor): Input image (B, 3, H, W) in [0, 1] or [-1, 1]
#         Returns:
#             Tensor: Probability of being a deepfake (B,)
#         """
#         # assert self.hific.model_mode == self.hific.args.ModelModes.EVALUATION
#         # assert not self.hific.training

#         # Manually extract latent and compress output
#         with torch.no_grad():
#             spatial_shape = tuple(x.size()[2:])
#             factor = 2 ** self.hific.Encoder.n_downsampling_layers
#             x_padded = self.hific.utils.pad_factor(x, x.size()[2:], factor)
#             y = self.hific.Encoder(x_padded)

#             hf_down = self.hific.Hyperprior.analysis_net.n_downsampling_layers
#             y_padded = self.hific.utils.pad_factor(y, y.size()[2:], 2 ** hf_down)

#             compression_output = self.hific.Hyperprior.compress_forward(y_padded, spatial_shape)

#         # Flatten latent y
#         y_flat = torch.flatten(y, start_dim=1)  # shape (B, latent_feat_dim)

#         if self.use_entropy_features:
#             bpp_tensor = torch.tensor([
#                 [compression_output.total_bpp, compression_output.latent_bpp, compression_output.hyperlatent_bpp]
#             ], dtype=torch.float32, device=x.device).repeat(x.size(0), 1)
#             features = torch.cat([y_flat, bpp_tensor], dim=1)
#         else:
#             features = y_flat

#         # Predict
#         prob = self.classifier(features).squeeze(1)
#         return prob

class HiFiCDeepfakeClassifier(nn.Module):
    def __init__(self, hific_model, use_entropy_features=True):
        """
        Args:
            hific_model (Model): Initialized HiFiC model (in eval mode).
            use_entropy_features (bool): Whether to append entropy stats as features.
        """
        super().__init__()
        self.hific = hific_model
        self.use_entropy_features = use_entropy_features
        self.tables_built = False  # NEW: flag to avoid rebuilding

        # Example latent dimensions (default: 320 x 32 x 32)
        dummy_input = torch.randn(1, 3, 256, 256)
        dummy_input = dummy_input.to(next(hific_model.parameters()).device)
        with torch.no_grad():
            y = self.hific.Encoder(dummy_input)
        latent_dim = y.numel()

        extra_dim = 3 if use_entropy_features else 0
        # in_features = latent_dim + extra_dim
        in_features = 220*16*16

        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels=220, out_channels=256, kernel_size=3), # (B, C=220, H=16, W=16) --> (B, C=256, H=14, W=14)
        #     nn.ReLU(),
        #     nn.Linear(in_features, 2048),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(2048),
        #     nn.Dropout(0.3),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(0.1),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid()
        # )
        self.classifier = nn.Sequential(
            nn.Conv2d(220, 64, kernel_size=3, padding=1),  # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 8, 8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (128, 1, 1)
            nn.Flatten(),  # (128,)
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image (B, 3, H, W) in [0,1] or [-1,1]
        Returns:
            Tensor: Deepfake probability per image (B,)
        """
        # assert self.hific.model_mode == self.hific.args.ModelModes.EVALUATION
        # assert not self.hific.training
        device = next(self.hific.parameters()).device
        x = x.to(device)
        # Build entropy tables if needed
        if not self.tables_built:
            self.hific.Hyperprior.hyperprior_entropy_model.build_tables()
            self.tables_built = True
        start_encoding = time.time()
        batch_size = x.size(0)
        features = []

        intermediates, hyperinfo, y = self.hific.forward(x)
        y_hat = intermediates.latents_quantized
        # y, compression_output = self.hific.compress(x, silent=True)
        # y_hat = self.hific.decompress(compression_output)
        
        features = y_hat - y # Latent residual
        end_encoding = time.time()
        
        prob = self.classifier(features).squeeze(1)  # (B,)
        end_cls = time.time()
        # print(f'Ellapsed time for encoding: {end_encoding-start_encoding}.')
        # print(f'Ellapsed time for classification: {end_cls-end_encoding}.')
        return prob
    
    













        # y_hat_flat = y_hat.view(y_hat.size(0), -1) # Flatten (1, C*H*W)

        # y = self.hific.Encoder(x)
        # y_flat = y.view(y.size(0), -1)  # Flatten (1, C*H*W)
        # print(f"y_hyperlatent={y_hat.shape} | y_latent={y.shape}")
        # if self.use_entropy_features:
            # bpp_feat = torch.tensor([compression_output.hyperlatents_encoded], device=device)
            # bpp_feat = bpp_feat.astype(np.float32)
            # bpp_tensor = torch.from_numpy(bpp_feat).to(device='cuda')
            # feature_vec = torch.cat([y_flat, bpp_tensor], dim=1)
        # else:
        #     feature_vec = y_flat

        # features.append(feature_vec)

        # features = torch.cat(features, dim=0)  # Shape: (B, feat_dim)