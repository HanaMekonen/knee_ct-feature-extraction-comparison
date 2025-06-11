#!/usr/bin/env python
# coding: utf-8

# ### 2. 2D-to-3D DenseNet121 Conversion

# In[7]:


# Load the pretrained 2D DenseNet121.
import torchvision
model2d = torchvision.models.densenet121(pretrained=True)
model2d.eval()


# In[19]:


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Configuration
DEPTH = 3   # Number of slices to inflate each 2D kernel across

# Helper functions for inflation 

def inflate_conv2d_to_3d(conv2d: nn.Conv2d, depth: int) -> nn.Conv3d:
    """Convert a 2D Conv to a 3D Conv by repeating its kernel along a new depth axis."""
    w2d = conv2d.weight.data
    k_h, k_w = conv2d.kernel_size
    conv3d = nn.Conv3d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=(depth, k_h, k_w),
        stride=(1, *conv2d.stride),
        padding=(depth // 2, *conv2d.padding),
        bias=(conv2d.bias is not None)
    )
    # Inflate weights: repeat & normalize
    w3d = w2d.unsqueeze(2).repeat(1, 1, depth, 1, 1).div_(depth)
    conv3d.weight.data.copy_(w3d)
    if conv2d.bias is not None:
        conv3d.bias.data.copy_(conv2d.bias.data)
    return conv3d

def convert_bn2d_to_3d(bn2d: nn.BatchNorm2d) -> nn.BatchNorm3d:
    """Convert BatchNorm2d to BatchNorm3d, copying parameters."""
    bn3d = nn.BatchNorm3d(
        num_features=bn2d.num_features,
        eps=bn2d.eps,
        momentum=bn2d.momentum,
        affine=bn2d.affine,
        track_running_stats=bn2d.track_running_stats
    )
    bn3d.weight.data.copy_(bn2d.weight.data)
    bn3d.bias.data.copy_(bn2d.bias.data)
    bn3d.running_mean.copy_(bn2d.running_mean)
    bn3d.running_var.copy_(bn2d.running_var)
    return bn3d

def convert_maxpool2d_to_3d(mp2d: nn.MaxPool2d) -> nn.MaxPool3d:
    """Convert MaxPool2d to MaxPool3d."""
    k = mp2d.kernel_size if isinstance(mp2d.kernel_size, tuple) else (mp2d.kernel_size,)*2
    s = mp2d.stride      if isinstance(mp2d.stride, tuple)      else (mp2d.stride,)*2
    p = mp2d.padding     if isinstance(mp2d.padding, tuple)     else (mp2d.padding,)*2
    return nn.MaxPool3d(kernel_size=(1, *k),
                        stride=(1, *s),
                        padding=(0, *p))

def convert_avgpool2d_to_3d(ap2d: nn.AvgPool2d) -> nn.AvgPool3d:
    """Convert AvgPool2d to AvgPool3d."""
    k = ap2d.kernel_size if isinstance(ap2d.kernel_size, tuple) else (ap2d.kernel_size,)*2
    s = ap2d.stride      if isinstance(ap2d.stride, tuple)      else (ap2d.stride,)*2
    p = ap2d.padding     if isinstance(ap2d.padding, tuple)     else (ap2d.padding,)*2
    return nn.AvgPool3d(kernel_size=(1, *k),
                        stride=(1, *s),
                        padding=(0, *p),
                        ceil_mode=ap2d.ceil_mode,
                        count_include_pad=ap2d.count_include_pad)

def convert_adaptivepool2d_to_3d(ap2d: nn.AdaptiveAvgPool2d) -> nn.AdaptiveAvgPool3d:
    """Convert AdaptiveAvgPool2d to AdaptiveAvgPool3d."""
    os = ap2d.output_size
    os = (os, os) if isinstance(os, int) else os
    return nn.AdaptiveAvgPool3d((1, *os))

#  Recursive in-place inflation 

def recursive_inflate_modules(module: nn.Module):
    """Walk through module and replace 2D layers with 3D equivalents."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, inflate_conv2d_to_3d(child, DEPTH))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, convert_bn2d_to_3d(child))
        elif isinstance(child, nn.MaxPool2d):
            setattr(module, name, convert_maxpool2d_to_3d(child))
        elif isinstance(child, nn.AvgPool2d):
            setattr(module, name, convert_avgpool2d_to_3d(child))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, convert_adaptivepool2d_to_3d(child))
        else:
            recursive_inflate_modules(child)

#  Build and convert DenseNet121

# 1. Load pretrained 2D DenseNet121
model2d = models.densenet121(pretrained=True)

# 2. Inflate all 2D layers in-place
recursive_inflate_modules(model2d)

# 3. Collapse the first conv to single-channel if your input is 1-channel
old_conv0 = model2d.features.conv0
out_ch, _, d, h, w = old_conv0.weight.shape
new_conv0 = nn.Conv3d(
    in_channels=1,
    out_channels=out_ch,
    kernel_size=old_conv0.kernel_size,
    stride=old_conv0.stride,
    padding=old_conv0.padding,
    bias=(old_conv0.bias is not None)
)
with torch.no_grad():
    collapsed_w = old_conv0.weight.mean(dim=1, keepdim=True)  # [out_ch,1,d,h,w]
    new_conv0.weight.copy_(collapsed_w)
    if old_conv0.bias is not None:
        new_conv0.bias.copy_(old_conv0.bias)
model2d.features.conv0 = new_conv0

# 4. Wrap with a custom forward to use 3D pooling
class DenseNet3D(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.features   = base_model.features
        self.classifier = base_model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                   # (B, C, D', H', W')
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool3d(x, (1,1,1))  # global 3D pooling
        x = x.view(x.size(0), -1)              # flatten
        return self.classifier(x)

model3d = DenseNet3D(model2d)

# Test with a dummy tensor 

dummy = torch.randn(1, 1, DEPTH, 224, 224)  # (batch, channels=1, depth, H, W)
output = model3d(dummy)
print("3D DenseNet121 output shape:", output.shape)



# In[ ]:


import torch
import os

print("Saving to:", os.getcwd())


save_path = "model3d_state_dict.pth"
torch.save(model3d.state_dict(), save_path)

print(f"Model state_dict saved to {save_path}")

