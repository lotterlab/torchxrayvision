"""
GAN Model Architectures for X-ray Windowing
- Generator: Hypernetwork outputting spline parameters
- Discriminator 1: Real vs Windowed image
- Discriminator 2: Dataset/scanner source classifier
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class MonotonicSplineWindowing(nn.Module):
    """
    Applies monotonic spline-based windowing to images.
    Based on the AutoStainer approach from the proposal.
    """
    def __init__(self, nbins=8):
        super().__init__()
        self.nbins = nbins
        
    def forward(self, images, spline_params):
        """
        Args:
            images: (B, C, H, W) input images in [0, 1]
            spline_params: (B, nbins) parameters for cubic spline
        Returns:
            windowed_images: (B, C, H, W) transformed images
        """
        B, C, H, W = images.shape
        
        # Create knot points for spline
        knots = torch.linspace(0, 1, self.nbins, device=images.device)
        
        # Ensure monotonicity: cumsum of softplus
        # This ensures the spline is monotonically increasing
        spline_values = torch.cumsum(torch.nn.functional.softplus(spline_params), dim=1)
        # Normalize to [0, 1]
        spline_values = spline_values / (spline_values[:, -1:] + 1e-8)
        
        # Flatten image for interpolation
        images_flat = images.reshape(B, C, -1)  # (B, C, H*W)
        
        # Apply spline interpolation
        windowed_flat = torch.zeros_like(images_flat)
        for b in range(B):
            for c in range(C):
                # Linear interpolation (can be replaced with cubic spline)
                windowed_flat[b, c] = torch.nn.functional.interpolate(
                    spline_values[b].unsqueeze(0).unsqueeze(0),
                    size=images_flat[b, c].shape[0],
                    mode='linear',
                    align_corners=True
                ).squeeze()
                
                # Map through the spline
                pixel_vals = images_flat[b, c]
                # Quantize to bin indices
                bin_indices = (pixel_vals * (self.nbins - 1)).long()
                bin_indices = torch.clamp(bin_indices, 0, self.nbins - 2)
                
                # Linear interpolation between knots
                t = pixel_vals * (self.nbins - 1) - bin_indices.float()
                windowed_flat[b, c] = spline_values[b, bin_indices] * (1 - t) + \
                                      spline_values[b, bin_indices + 1] * t
        
        windowed_images = windowed_flat.reshape(B, C, H, W)
        return windowed_images


class Generator(nn.Module):
    """
    Generator/Hypernetwork that outputs spline parameters for windowing.
    """
    def __init__(self, backbone='resnet50', window_nbins=8, input_channels=1):
        super().__init__()
        self.window_nbins = window_nbins
        
        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=False)
            # Modify first conv for single channel
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                        stride=2, padding=3, bias=False)
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 2048
            
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=False)
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                        stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 512
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Spline parameter head
        self.spline_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, window_nbins)
        )
        
        self.windowing = MonotonicSplineWindowing(nbins=window_nbins)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            windowed: (B, C, H, W) windowed images
            spline_params: (B, nbins) spline parameters
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Generate spline parameters
        spline_params = self.spline_head(features)
        
        # Apply windowing
        windowed = self.windowing(x, spline_params)
        
        return windowed, spline_params


class RealFakeDiscriminator(nn.Module):
    """
    Discriminator to distinguish real from windowed images.
    """
    def __init__(self, architecture='resnet18', input_channels=1):
        super().__init__()
        
        if architecture == 'resnet18':
            base_model = models.resnet18(pretrained=False)
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 512
            
        elif architecture == 'resnet50':
            base_model = models.resnet50(pretrained=False)
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 2048
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) images
        Returns:
            prob: (B, 1) probability of being real
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        prob = self.classifier(features)
        return prob


class DatasetDiscriminator(nn.Module):
    """
    Discriminator to classify which dataset/scanner an image comes from.
    This is the adversarial component - generator tries to fool this.
    """
    def __init__(self, architecture='resnet18', num_datasets=2, input_channels=1):
        super().__init__()
        self.num_datasets = num_datasets
        
        if architecture == 'resnet18':
            base_model = models.resnet18(pretrained=False)
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 512
            
        elif architecture == 'resnet50':
            base_model = models.resnet50(pretrained=False)
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 2048
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Multi-class classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_datasets)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) images
        Returns:
            logits: (B, num_datasets) class logits
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


class SpectralNorm(nn.Module):
    """
    Spectral normalization for stable GAN training.
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))


def weights_init(m):
    """
    Initialize model weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)