import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import WeightedRandomSampler
import numpy as np
from pathlib import Path
import time
import copy
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import seaborn as sns
import random

# Set cache directory to /medias instead of home directory
os.environ['TORCH_HOME'] = '../models/mixed/model_cache'
# os.environ['TORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def extract_chrominance_features(image):
    """Extract chrominance features from RGB image"""
    try:
        # Convert to YUV color space
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.shape[0] == 3:  # If image is in CHW format
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure image is in uint8 format and in valid range [0, 255]
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Check if image is valid
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Invalid image dimensions")
        
        # Convert to different color spaces for more comprehensive analysis
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Extract chrominance channels
        u_channel = yuv[:, :, 1]        
        v_channel = yuv[:, :, 2]
        
        # Extract saturation and hue from HSV
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        
        # Extract a and b from LAB
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Calculate basic statistics for each channel
        u_mean, u_std = np.mean(u_channel), np.std(u_channel)
        v_mean, v_std = np.mean(v_channel), np.std(v_channel)
        h_mean, h_std = np.mean(h_channel), np.std(h_channel)
        s_mean, s_std = np.mean(s_channel), np.std(s_channel)
        a_mean, a_std = np.mean(a_channel), np.std(a_channel)
        b_mean, b_std = np.mean(b_channel), np.std(b_channel)
        
        # Calculate color coherence
        # Measure the uniformity/consistency of colors
        u_coherence = np.percentile(u_channel, 75) - np.percentile(u_channel, 25)
        v_coherence = np.percentile(v_channel, 75) - np.percentile(v_channel, 25)
        
        # Calculate image regions color consistency
        # Divided the image into 4 quadrants and calculate the standard deviation of means
        height, width = u_channel.shape
        mid_h, mid_w = height // 2, width // 2
        
        u_regions = [
            u_channel[:mid_h, :mid_w],
            u_channel[:mid_h, mid_w:],
            u_channel[mid_h:, :mid_w],
            u_channel[mid_h:, mid_w:]
        ]
        
        v_regions = [
            v_channel[:mid_h, :mid_w],
            v_channel[:mid_h, mid_w:],
            v_channel[mid_h:, :mid_w],
            v_channel[mid_h:, mid_w:]
        ]
        
        u_region_means = np.array([np.mean(region) for region in u_regions])
        v_region_means = np.array([np.mean(region) for region in v_regions])
        
        u_region_consistency = np.std(u_region_means)
        v_region_consistency = np.std(v_region_means)
        
        # Calculate color moments (mean, variance, skewness, kurtosis)
        def color_moments(channel):
            try:
                mean = np.mean(channel)
                if isinstance(mean, np.ndarray):
                    mean = mean.item()
                
                variance = np.var(channel)
                if isinstance(variance, np.ndarray):
                    variance = variance.item()
                
                # Calculate skewness
                if variance > 0:
                    skewness = np.mean(((channel - mean) / np.sqrt(variance)) ** 3)
                    if isinstance(skewness, np.ndarray):
                        skewness = skewness.item()
                else:
                    skewness = 0
                
                # Calculate kurtosis
                if variance > 0:
                    kurtosis = np.mean(((channel - mean) / np.sqrt(variance)) ** 4)
                    if isinstance(kurtosis, np.ndarray):
                        kurtosis = kurtosis.item()
                else:
                    kurtosis = 0
                
                return [mean, variance, skewness, kurtosis]
            except Exception as e:
                print(f"Error in color_moments: {str(e)}")
                return [0, 0, 0, 0]  # Return default values in case of error
        
        # Calculate color moments for each channel
        u_moments = color_moments(u_channel)
        v_moments = color_moments(v_channel)
        h_moments = color_moments(h_channel)
        s_moments = color_moments(s_channel)
        a_moments = color_moments(a_channel)
        b_moments = color_moments(b_channel)
        
        # SIMPLIFIED APPROACH: Use histogram-based features instead of CCV
        # This avoids the problematic connected components calculation
        def calculate_histogram_features(channel, bins=10):
            try:
                # Normalize channel to [0,1]
                channel_norm = (channel - np.min(channel)) / (np.max(channel) - np.min(channel) + 1e-10)
                
                # Create a histogram
                hist, _ = np.histogram(channel_norm, bins=bins, range=(0, 1))
                
                # Normalize histogram
                hist = hist / (np.sum(hist) + 1e-10)
                
                # Calculate histogram statistics
                mean = np.mean(hist)
                std = np.std(hist)
                
                # Calculate histogram shape features
                # These approximate the coherence concept without using connected components
                left_half = np.sum(hist[:bins//2])
                right_half = np.sum(hist[bins//2:])
                center_weight = np.sum(hist[bins//3:2*bins//3])
                
                return [mean, std, left_half, right_half, center_weight]
            except Exception as e:
                print(f"Error in calculate_histogram_features: {str(e)}")
                return [0] * 5  # Return default values in case of error
        
        # Calculate histogram features for each channel
        u_hist_features = calculate_histogram_features(u_channel)
        v_hist_features = calculate_histogram_features(v_channel)
        
        # Combine all features
        features = [
            u_mean, v_mean, u_std, v_std,
            h_mean, s_mean, h_std, s_std,
            a_mean, b_mean, a_std, b_std,
            u_coherence, v_coherence,
            u_region_consistency, v_region_consistency
        ]
        
        # Add color moments
        features.extend(u_moments)
        features.extend(v_moments)
        features.extend(h_moments)
        features.extend(s_moments)
        features.extend(a_moments)
        features.extend(b_moments)
        
        # Add histogram features instead of CCV
        features.extend(u_hist_features)
        features.extend(v_hist_features)
        
        return torch.tensor(features, dtype=torch.float32)
    except Exception as e:
        # Return a zero tensor with the expected size in case of error
        # This ensures the training can continue even if feature extraction fails
        print(f"Error in extract_chrominance_features: {str(e)}")
        # Return a zero tensor with the expected size (adjust size as needed)
        return torch.zeros(50, dtype=torch.float32)  # Adjust size based on your feature vector length

def compute_color_histograms(image, bins=32):
    """Extract color histogram features from image"""
    try:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.shape[0] == 3:  # If image is in CHW format
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure image is in uint8 format and in valid range [0, 255]
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Check if image is valid
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Invalid image dimensions")
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Compute histograms for HSV
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Compute histograms for LAB
        l_hist = cv2.calcHist([lab], [0], None, [bins], [0, 256])
        a_hist = cv2.calcHist([lab], [1], None, [bins], [0, 256])
        b_hist = cv2.calcHist([lab], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        l_hist = cv2.normalize(l_hist, l_hist).flatten()
        a_hist = cv2.normalize(a_hist, a_hist).flatten()
        b_hist = cv2.normalize(b_hist, b_hist).flatten()
        
        # Calculate histogram statistics
        def hist_stats(hist):
            try:
                mean = np.mean(hist)
                if isinstance(mean, np.ndarray):
                    mean = mean.item()
                
                std = np.std(hist)
                if isinstance(std, np.ndarray):
                    std = std.item()
                
                # Calculate skewness
                if std > 0:
                    skewness = np.mean(((hist - mean) / (std + 1e-10)) ** 3)
                    if isinstance(skewness, np.ndarray):
                        skewness = skewness.item()
                else:
                    skewness = 0
                
                # Calculate kurtosis
                if std > 0:
                    kurtosis = np.mean(((hist - mean) / (std + 1e-10)) ** 4)
                    if isinstance(kurtosis, np.ndarray):
                        kurtosis = kurtosis.item()
                else:
                    kurtosis = 0
                
                return [mean, std, skewness, kurtosis]
            except Exception as e:
                print(f"Error in hist_stats: {str(e)}")
                return [0, 0, 0, 0]  # Return default values in case of error
        
        h_stats = hist_stats(h_hist)
        s_stats = hist_stats(s_hist)
        v_stats = hist_stats(v_hist)
        l_stats = hist_stats(l_hist)
        a_stats = hist_stats(a_hist)
        b_stats = hist_stats(b_hist)
        
        # Combine histograms and stats into a single feature vector
        hist_features = np.concatenate([
            h_hist, s_hist, v_hist, l_hist, a_hist, b_hist,
            h_stats, s_stats, v_stats, l_stats, a_stats, b_stats
        ])
        
        return torch.tensor(hist_features, dtype=torch.float32)
    except Exception as e:
        # Return a zero tensor with the expected size in case of error
        print(f"Error in compute_color_histograms: {str(e)}")
        # Return a zero tensor with the expected size (adjust size as needed)
        # Size = bins*6 (histograms) + 4*6 (stats)
        return torch.zeros(bins*6 + 24, dtype=torch.float32)

class ChrominanceLoss(nn.Module):
    def __init__(self, chroma_weight=0.05, hist_weight=0.02):
        super(ChrominanceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.chroma_weight = chroma_weight
        self.hist_weight = hist_weight
        self.mse = nn.MSELoss()
        
    def forward(self, outputs, labels, images, color_features=None):
        # Classification loss
        cls_loss = self.ce_loss(outputs, labels)
        
        # Initialize loss components
        chrom_loss = torch.tensor(0.0, device=outputs.device)
        hist_loss = torch.tensor(0.0, device=outputs.device)
        
        try:
            # Separate fake and real images based on labels
            fake_indices = (labels == 0).nonzero(as_tuple=True)[0]
            real_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            # Skip if we don't have both fake and real images in the batch
            if len(fake_indices) > 0 and len(real_indices) > 0:
                fake_features = []
                real_features = []
                fake_hists = []
                real_hists = []
                
                # Extract features for all images
                for i in range(images.size(0)):
                    try:
                        img_chrom_features = extract_chrominance_features(images[i])
                        img_hist_features = compute_color_histograms(images[i])
                        
                        if i in fake_indices:
                            fake_features.append(img_chrom_features)
                            fake_hists.append(img_hist_features)
                        else:
                            real_features.append(img_chrom_features)
                            real_hists.append(img_hist_features)
                    except Exception as e:
                        print(f"Warning: Failed to extract features for image {i}: {str(e)}")
                        continue
                
                # Convert lists to tensors if we have features
                if fake_features and real_features:
                    try:
                        fake_features_tensor = torch.stack(fake_features).to(outputs.device)
                        real_features_tensor = torch.stack(real_features).to(outputs.device)
                        
                        # Get mean features for fake and real
                        fake_mean = torch.mean(fake_features_tensor, dim=0)
                        real_mean = torch.mean(real_features_tensor, dim=0)
                        
                        # Maximize distance between fake and real features (contrastive approach)
                        chrom_loss = -self.mse(fake_mean, real_mean)
                        
                        # Do the same for histogram features
                        if fake_hists and real_hists:
                            fake_hists_tensor = torch.stack(fake_hists).to(outputs.device)
                            real_hists_tensor = torch.stack(real_hists).to(outputs.device)
                            
                            fake_hist_mean = torch.mean(fake_hists_tensor, dim=0)
                            real_hist_mean = torch.mean(real_hists_tensor, dim=0)
                            
                            hist_loss = -self.mse(fake_hist_mean, real_hist_mean)
                    except Exception as e:
                        print(f"Warning: Failed to compute feature losses: {str(e)}")
        except Exception as e:
            print(f"Warning: Error in ChrominanceLoss forward pass: {str(e)}")
        
        # Combine losses with weights
        total_loss = cls_loss + self.chroma_weight * chrom_loss + self.hist_weight * hist_loss
        
        # Return individual loss components for monitoring
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'chrom_loss': chrom_loss.item(),
            'hist_loss': hist_loss.item(),
            'total_loss': total_loss.item()
        }

class ColorAwareResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ColorAwareResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        
        # Simplified color processing branch with reduced parameters
        self.color_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simplified texture analysis branch
        self.texture_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simplified chrominance analysis branch
        self.chroma_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),  # 1x1 convolutions to focus on color
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simplified color histogram branch
        self.hist_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4),  # Large kernel for global color stats
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simplified color coherence branch
        self.coherence_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),  # Large kernel to capture color coherence
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simplified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 64 + 64 + 64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 6),  # 6 attention weights for each branch
            nn.Softmax(dim=1)
        )
        
        # Simplified color attention mechanism
        self.color_attention = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 64 + 64 + 64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 6),  # 6 attention weights for each branch
            nn.Softmax(dim=1)
        )
        
        # Simplified final classification layers
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 64 + 64 + 64 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Simplified color feature extractor
        self.color_feature_extractor = nn.Sequential(
            nn.Linear(num_ftrs + 128 + 64 + 64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Get features from each branch
        # ResNet features
        resnet_features = self.resnet.conv1(x)
        resnet_features = self.resnet.bn1(resnet_features)
        resnet_features = self.resnet.relu(resnet_features)
        resnet_features = self.resnet.maxpool(resnet_features)
        
        resnet_features = self.resnet.layer1(resnet_features)
        resnet_features = self.resnet.layer2(resnet_features)
        resnet_features = self.resnet.layer3(resnet_features)
        resnet_features = self.resnet.layer4(resnet_features)
        
        resnet_features = self.resnet.avgpool(resnet_features)
        resnet_features = torch.flatten(resnet_features, 1)  # [batch_size, num_ftrs]
        
        # Color branch features
        color_features = self.color_branch(x)
        color_features = torch.flatten(color_features, 1)  # [batch_size, 128]
        
        # Texture branch features
        texture_features = self.texture_branch(x)
        texture_features = torch.flatten(texture_features, 1)  # [batch_size, 64]
        
        # Chrominance branch features
        chroma_features = self.chroma_branch(x)
        chroma_features = torch.flatten(chroma_features, 1)  # [batch_size, 64]
        
        # Histogram branch features
        hist_features = self.hist_branch(x)
        hist_features = torch.flatten(hist_features, 1)  # [batch_size, 64]
        
        # Coherence branch features
        coherence_features = self.coherence_branch(x)
        coherence_features = torch.flatten(coherence_features, 1)  # [batch_size, 64]
        
        # Stack features separately
        features_list = [resnet_features, color_features, texture_features, 
                         chroma_features, hist_features, coherence_features]
        
        # Get attention weights
        combined_features = torch.cat(features_list, dim=1)
        attention_weights = self.attention(combined_features)  # [batch_size, 6]
        
        # Get color-specific attention weights
        color_attention_weights = self.color_attention(combined_features)  # [batch_size, 6]
        
        # Apply attention weights correctly
        weighted_features = []
        for i, features in enumerate(features_list):
            weighted_features.append(features * attention_weights[:, i:i+1])
        
        # Apply color-specific attention to color-related branches
        color_weighted_features = []
        color_branches = [color_features, chroma_features, hist_features, coherence_features]
        for i, features in enumerate(color_branches):
            color_weighted_features.append(features * color_attention_weights[:, i+1:i+2])
        
        # Combine weighted features
        weighted_combined = torch.cat(weighted_features, dim=1)
        
        # Combine color-weighted features
        color_weighted_combined = torch.cat(color_weighted_features, dim=1)
        
        # Extract color features for visualization if needed
        color_representation = self.color_feature_extractor(weighted_combined)
        
        # Final classification
        output = self.fc(weighted_combined)
        
        # Return both output and color features
        return output, color_representation

def visualize_attention_weights(model, dataloader, device, save_dir='/medias/db/ImagingSecurity_misc/elmennao/models/visualizations'):
    """Visualize the attention weights to understand which features are most important"""
    model.eval()
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect attention weights
    all_attention_weights = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # Forward pass to get attention weights
            _, color_features = model(inputs)
            
            # Get attention weights from the model
            combined_features = torch.cat([
                model.resnet.avgpool(model.resnet.layer4(model.resnet.layer3(model.resnet.layer2(model.resnet.layer1(
                    model.resnet.maxpool(model.resnet.relu(model.resnet.bn1(model.resnet.conv1(inputs))))
                ))))).flatten(1),
                model.color_branch(inputs).flatten(1),
                model.texture_branch(inputs).flatten(1),
                model.chroma_branch(inputs).flatten(1),
                model.hist_branch(inputs).flatten(1),
                model.coherence_branch(inputs).flatten(1)
            ], dim=1)
            
            attention_weights = model.attention(combined_features)
            
            all_attention_weights.append(attention_weights.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if len(all_attention_weights) >= 50:  # Limit to 50 batches
                break
    
    # Concatenate all batches
    all_attention_weights = np.concatenate(all_attention_weights, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate average attention weights for real and fake images
    fake_indices = all_labels == 0
    real_indices = all_labels == 1
    
    fake_weights = all_attention_weights[fake_indices]
    real_weights = all_attention_weights[real_indices]
    
    fake_mean_weights = np.mean(fake_weights, axis=0)
    real_mean_weights = np.mean(real_weights, axis=0)
    
    # Plot attention weights
    feature_names = ['ResNet', 'Color', 'Texture', 'Chroma', 'Histogram', 'Coherence']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, fake_mean_weights, width, label='Fake')
    plt.bar(x + width/2, real_mean_weights, width, label='Real')
    
    plt.xlabel('Feature Branch')
    plt.ylabel('Attention Weight')
    plt.title('Average Attention Weights by Image Class')
    plt.xticks(x, feature_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_weights.png'))
    plt.close()
    
    print(f"Visualization saved to {os.path.join(save_dir, 'attention_weights.png')}")

def visualize_color_features(model, dataloader, device, save_dir='/medias/db/ImagingSecurity_misc/elmennao/models/visualizations'):
    """Visualize the learned color features using t-SNE"""
    model.eval()
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect color features and labels
    all_color_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # Forward pass to get color features
            _, color_features = model(inputs)
            
            all_color_features.append(color_features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if len(all_color_features) >= 50:  # Limit to 50 batches for speed
                break
    
    # Concatenate all batches
    all_color_features = np.concatenate(all_color_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    color_features_2d = tsne.fit_transform(all_color_features)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        color_features_2d[:, 0], 
        color_features_2d[:, 1], 
        c=all_labels, 
        cmap='coolwarm', 
        alpha=0.7
    )
    
    plt.colorbar(scatter, label='Class (0=Fake, 1=Real)')
    plt.title('t-SNE Visualization of Color Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'color_features_tsne.png'))
    plt.close()
    
    # Also try PCA visualization
    pca = PCA(n_components=2)
    color_features_pca = pca.fit_transform(all_color_features)
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        color_features_pca[:, 0], 
        color_features_pca[:, 1], 
        c=all_labels, 
        cmap='coolwarm', 
        alpha=0.7
    )
    
    plt.colorbar(scatter, label='Class (0=Fake, 1=Real)')
    plt.title('PCA Visualization of Color Features')
    plt.xlabel('PC1 ({}% Variance)'.format(round(pca.explained_variance_ratio_[0] * 100, 2)))
    plt.ylabel('PC2 ({}% Variance)'.format(round(pca.explained_variance_ratio_[1] * 100, 2)))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'color_features_pca.png'))
    plt.close()
    
    print(f"Visualizations saved to {save_dir}")

def calculate_metrics(outputs, labels):
    """Calculate accuracy for each class"""
    # For our enhanced model, outputs is a tuple (predictions, color_features)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    _, preds = torch.max(outputs, 1)
    
    # Calculate metrics for each class
    fake_mask = labels == 0
    real_mask = labels == 1
    
    fake_correct = torch.sum((preds == labels)[fake_mask]).float()
    real_correct = torch.sum((preds == labels)[real_mask]).float()
    
    fake_total = torch.sum(fake_mask).float()
    real_total = torch.sum(real_mask).float()
    
    fake_acc = fake_correct / fake_total if fake_total > 0 else torch.tensor(0.0)
    real_acc = real_correct / real_total if real_total > 0 else torch.tensor(0.0)
    
    return {
        'fake_correct': fake_correct,
        'real_correct': real_correct,
        'fake_total': fake_total,
        'real_total': real_total,
        'fake_acc': fake_acc,
        'real_acc': real_acc
    }

def create_weighted_sampler(dataset):
    """Create a weighted sampler to handle class imbalance"""
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, labels):
        # Ensure outputs and labels have the same batch size
        if outputs.size(0) != labels.size(0):
            raise ValueError(f"Expected outputs batch_size ({outputs.size(0)}) to match labels batch_size ({labels.size(0)})")
            
        ce_loss = self.ce_loss(outputs, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Apply mixup to the criterion"""
    # Check if criterion is FocalLoss
    if isinstance(criterion, FocalLoss):
        # For FocalLoss, we need to handle the mixup differently
        # since it doesn't support reduction='none' like CrossEntropyLoss
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    else:
        # For other loss functions that support reduction='none'
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class DomainAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        self.color_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
        ])
        
    def __call__(self, img):
        if random.random() < self.p:
            return self.color_transforms(img)
        return img

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=250, visualize_every=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Set up visualizations directory
    vis_dir = '/medias/db/ImagingSecurity_misc/elmennao/models/visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create focal loss for better handling of class imbalance
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    
    # Create domain augmentation
    domain_aug = DomainAugmentation(p=0.5)
    
    # Create a log file to track training progress
    log_file = os.path.join(vis_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset sizes: {dataset_sizes}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write("-" * 50 + "\n")

    # Set gradient accumulation steps to simulate larger batch size
    gradient_accumulation_steps = 2

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}\n')
            f.write('-' * 50 + '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_metrics = {
                'fake_correct': 0,
                'real_correct': 0,
                'fake_total': 0,
                'real_total': 0
            }
            
            # Track individual loss components
            running_loss_components = {
                'cls_loss': 0.0,
                'chrom_loss': 0.0,
                'hist_loss': 0.0,
                'total_loss': 0.0
            }

            # Iterate over data
            optimizer.zero_grad()  # Zero gradients at the start of each epoch
            
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Apply domain augmentation during training
                if phase == 'train':
                    # Apply mixup with 30% probability
                    if random.random() < 0.3:
                        # Apply mixup to inputs and labels
                        mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
                        use_mixup = True
                    else:
                        mixed_inputs = inputs
                        use_mixup = False
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass with mixed inputs if using mixup
                    if use_mixup:
                        outputs, color_features = model(mixed_inputs)
                    else:
                        outputs, color_features = model(inputs)
                    
                    if use_mixup:
                        # Use mixup criterion with focal loss
                        loss = mixup_criterion(focal_loss, outputs, labels_a, labels_b, lam)
                        loss_components = {
                            'cls_loss': loss.item(),
                            'chrom_loss': 0.0,
                            'hist_loss': 0.0,
                            'total_loss': loss.item()
                        }
                    else:
                        # Use regular criterion with color features
                        loss, loss_components = criterion(outputs, labels, inputs)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                    if phase == 'train':
                        loss.backward()
                        
                        # Only step optimizer after accumulating gradients
                        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(dataloaders[phase]):
                            # Add gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()  # Zero gradients after step

                # Statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size * gradient_accumulation_steps
                
                # Update loss components
                for key in running_loss_components:
                    running_loss_components[key] += loss_components[key] * batch_size
                
                # Calculate detailed metrics
                metrics = calculate_metrics((outputs, color_features), labels)
                running_metrics['fake_correct'] += metrics['fake_correct']
                running_metrics['real_correct'] += metrics['real_correct']
                running_metrics['fake_total'] += metrics['fake_total']
                running_metrics['real_total'] += metrics['real_total']
                
                # Print batch progress
                if (i + 1) % 10 == 0:
                    print(f'Batch {i+1}/{len(dataloaders[phase])}')
                    # Print memory usage
                    if torch.cuda.is_available():
                        print(f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated')
                
                # Clear cache periodically to free memory
                if (i + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            
            # Calculate average loss components
            avg_loss_components = {k: v / dataset_sizes[phase] for k, v in running_loss_components.items()}
            
            # Calculate epoch metrics
            fake_acc = (running_metrics['fake_correct'] / running_metrics['fake_total']).item() if running_metrics['fake_total'] > 0 else 0
            real_acc = (running_metrics['real_correct'] / running_metrics['real_total']).item() if running_metrics['real_total'] > 0 else 0
            total_acc = ((running_metrics['fake_correct'] + running_metrics['real_correct']) / 
                        (running_metrics['fake_total'] + running_metrics['real_total'])).item()

            print(f'{phase} Metrics:')
            print(f'Loss: {epoch_loss:.4f}')
            print(f'Fake Accuracy: {fake_acc:.4f} ({running_metrics["fake_correct"]:.0f}/{running_metrics["fake_total"]:.0f})')
            print(f'Real Accuracy: {real_acc:.4f} ({running_metrics["real_correct"]:.0f}/{running_metrics["real_total"]:.0f})')
            print(f'Total Accuracy: {total_acc:.4f}')
            
            # Print loss components
            print(f'Loss Components:')
            for key, value in avg_loss_components.items():
                print(f'  {key}: {value:.4f}')
            
            print('-' * 50)
            
            # Log metrics to file
            with open(log_file, 'a') as f:
                f.write(f'{phase} Metrics:\n')
                f.write(f'Loss: {epoch_loss:.4f}\n')
                f.write(f'Fake Accuracy: {fake_acc:.4f} ({running_metrics["fake_correct"]:.0f}/{running_metrics["fake_total"]:.0f})\n')
                f.write(f'Real Accuracy: {real_acc:.4f} ({running_metrics["real_correct"]:.0f}/{running_metrics["real_total"]:.0f})\n')
                f.write(f'Total Accuracy: {total_acc:.4f}\n')
                f.write(f'Loss Components:\n')
                for key, value in avg_loss_components.items():
                    f.write(f'  {key}: {value:.4f}\n')
                f.write('-' * 50 + '\n')

            # Deep copy the model if best accuracy
            if phase == 'val' and total_acc > best_acc:
                best_acc = total_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save(model.state_dict(), '/medias/db/ImagingSecurity_misc/elmennao/models/colorbest_deepfake_detector.pth')
        
        # Visualize color features every few epochs
        if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
            print("Generating visualizations...")
            try:
                visualize_attention_weights(model, dataloaders['val'], device, save_dir=f'{vis_dir}/epoch_{epoch+1}')
                visualize_color_features(model, dataloaders['val'], device, save_dir=f'{vis_dir}/epoch_{epoch+1}')
            except Exception as e:
                print(f"Error during visualization: {str(e)}")
            
        print()
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    with open(log_file, 'a') as f:
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        f.write(f'Best val Acc: {best_acc:4f}\n')

    # Generate final visualizations
    print("Generating final visualizations...")
    try:
        visualize_attention_weights(model, dataloaders['val'], device, save_dir=f'{vis_dir}/final')
        visualize_color_features(model, dataloaders['val'], device, save_dir=f'{vis_dir}/final')
    except Exception as e:
        print(f"Error during visualization: {str(e)}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = torch.ones(len(models)) / len(models)
        self.weights = nn.Parameter(weights)
        
    def forward(self, x):
        outputs = []
        color_features_list = []
        
        for model in self.models:
            output, color_features = model(x)
            outputs.append(output)
            color_features_list.append(color_features)
            
        # Stack and weight the outputs
        outputs = torch.stack(outputs)
        weighted_outputs = outputs * self.weights.view(-1, 1, 1)
        final_output = torch.sum(weighted_outputs, dim=0)
        
        # Average the color features
        color_features = torch.stack(color_features_list)
        final_color_features = torch.mean(color_features, dim=0)
        
        return final_output, final_color_features

class AdvancedDataAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        self.color_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.3),
            transforms.RandomPosterize(bits=4, p=0.3),
        ])
        
        self.geometric_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomRotation(degrees=15),
        ])
        
        # Move noise transforms after ToTensor
        self.noise_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3),
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            ], p=0.3),
        ])
        
    def __call__(self, img):
        if random.random() < self.p:
            img = self.color_transforms(img)
            img = self.geometric_transforms(img)
            # Noise transforms will be applied after ToTensor in the main transform pipeline
        return img

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv1(x)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class EnhancedChrominanceLoss(nn.Module):
    def __init__(self, chroma_weight=0.05, hist_weight=0.02, attention_weight=0.03):
        super(EnhancedChrominanceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.chroma_weight = chroma_weight
        self.hist_weight = hist_weight
        self.attention_weight = attention_weight
        self.mse = nn.MSELoss()
        
    def forward(self, outputs, labels, images, color_features=None):
        # Classification loss
        cls_loss = self.ce_loss(outputs, labels)
        
        # Initialize loss components
        chrom_loss = torch.tensor(0.0, device=outputs.device)
        hist_loss = torch.tensor(0.0, device=outputs.device)
        attention_loss = torch.tensor(0.0, device=outputs.device)
        
        try:
            # Separate fake and real images based on labels
            fake_indices = (labels == 0).nonzero(as_tuple=True)[0]
            real_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            if len(fake_indices) > 0 and len(real_indices) > 0:
                fake_features = []
                real_features = []
                fake_hists = []
                real_hists = []
                
                for i in range(images.size(0)):
                    try:
                        img_chrom_features = extract_chrominance_features(images[i])
                        img_hist_features = compute_color_histograms(images[i])
                        
                        if i in fake_indices:
                            fake_features.append(img_chrom_features)
                            fake_hists.append(img_hist_features)
                        else:
                            real_features.append(img_chrom_features)
                            real_hists.append(img_hist_features)
                    except Exception as e:
                        print(f"Warning: Failed to extract features for image {i}: {str(e)}")
                        continue
                
                if fake_features and real_features:
                    fake_features_tensor = torch.stack(fake_features).to(outputs.device)
                    real_features_tensor = torch.stack(real_features).to(outputs.device)
                    
                    # Enhanced feature comparison
                    fake_mean = torch.mean(fake_features_tensor, dim=0)
                    real_mean = torch.mean(real_features_tensor, dim=0)
                    
                    # Calculate cosine similarity between mean features
                    similarity_between_means = torch.cosine_similarity(fake_mean.unsqueeze(0), real_mean.unsqueeze(0), dim=1)
                    # We want to MINIMIZE similarity (maximize distance), so loss is -similarity
                    attention_loss = -similarity_between_means.mean()
                    
                    # MSE loss between means (maximize distance -> minimize negative MSE)
                    chrom_loss = -self.mse(fake_mean, real_mean)
                    
                    if fake_hists and real_hists:
                        fake_hists_tensor = torch.stack(fake_hists).to(outputs.device)
                        real_hists_tensor = torch.stack(real_hists).to(outputs.device)
                        
                        fake_hist_mean = torch.mean(fake_hists_tensor, dim=0)
                        real_hist_mean = torch.mean(real_hists_tensor, dim=0)
                        
                        # Maximize distance between histogram means
                        hist_loss = -self.mse(fake_hist_mean, real_hist_mean)
        except Exception as e:
            print(f"Warning: Error in EnhancedChrominanceLoss forward pass: {str(e)}")
        
        total_loss = (cls_loss + 
                     self.chroma_weight * chrom_loss + 
                     self.hist_weight * hist_loss + 
                     self.attention_weight * attention_loss)
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'chrom_loss': chrom_loss.item(),
            'hist_loss': hist_loss.item(),
            'attention_loss': attention_loss.item(),
            'total_loss': total_loss.item()
        }

def main():
    # Create cache directory if it doesn't exist
    cache_dir = '/medias/db/ImagingSecurity_misc/elmennao/model_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set memory optimization settings
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Enhanced data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            AdvancedDataAugmentation(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # Add noise transforms after ToTensor
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3),
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            ], p=0.3),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = Path("/medias/db/ImagingSecurity_misc/elmennao/dataset")
    
    # Create datasets
    image_datasets = {
        'train': datasets.ImageFolder(data_dir / 'train', data_transforms['train']),
        'val': datasets.ImageFolder(data_dir / 'test', data_transforms['val'])
    }
    
    # Create weighted sampler for training data
    train_sampler = create_weighted_sampler(image_datasets['train'])
    
    # Create dataloaders with reduced batch size to prevent memory issues
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=16, sampler=train_sampler, num_workers=2
        ),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'], batch_size=16, shuffle=False, num_workers=2
        )
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Dataset sizes: {dataset_sizes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print GPU memory info
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Available GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")

    # Create multiple models for ensemble
    models = [
        ColorAwareResNet(num_classes=2),
        ColorAwareResNet(num_classes=2),
        ColorAwareResNet(num_classes=2)
    ]
    
    # Create ensemble model
    ensemble_model = EnsembleModel(models)
    ensemble_model = ensemble_model.to(device)
    
    # Use enhanced loss function
    criterion = EnhancedChrominanceLoss(
        chroma_weight=0.05,
        hist_weight=0.02,
        attention_weight=0.03
    )
    
    # Simplified optimizer configuration
    optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Use CosineAnnealingWarmRestarts instead of OneCycleLR
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Reset LR every 10 epochs
        T_mult=2,  # Double the reset period after each restart
        eta_min=1e-6  # Minimum learning rate
    )

    # Set output directory for models and visualizations
    output_dir = '/medias/db/ImagingSecurity_misc/elmennao/models'
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Train the ensemble model
    model = train_model(ensemble_model, criterion, optimizer, scheduler, dataloaders, 
                      dataset_sizes, device, num_epochs=250, visualize_every=10)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'ensemble_deepfake_detector.pth'))
    print("Training completed and model saved!")
    
    # Generate final visualizations
    print("Generating final visualizations...")
    visualize_attention_weights(model, dataloaders['val'], device)
    visualize_color_features(model, dataloaders['val'], device)

if __name__ == '__main__':
    main() 


# gooooooooooooooood dataset for transfer  
# https://github.com/victorkitov/style-transfer-dataset 
