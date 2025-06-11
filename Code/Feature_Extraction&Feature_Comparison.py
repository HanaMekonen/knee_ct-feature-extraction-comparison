#!/usr/bin/env python
# coding: utf-8

# ###  Feature Extraction and Feature Comparison

# In[10]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from torchvision import models

# Configuration
DEPTH = 3
DEVICE = torch.device("cpu")  # CPU-only system
MODEL_WEIGHTS = "model3d_state_dict.pth"
CT_PATH = "3702_left_knee.nii.gz"
MASK_PATH = "bone_segmented.nii.gz"

# --- Model Conversion Utilities ---
def inflate_conv2d_to_3d(conv2d, depth):
    w2d = conv2d.weight.data
    k_h, k_w = conv2d.kernel_size
    conv3d = nn.Conv3d(
        conv2d.in_channels, conv2d.out_channels,
        kernel_size=(depth, k_h, k_w),
        stride=(1, *conv2d.stride),
        padding=(depth // 2, *conv2d.padding),
        bias=conv2d.bias is not None
    )
    w3d = w2d.unsqueeze(2).repeat(1, 1, depth, 1, 1).div_(depth)
    conv3d.weight.data.copy_(w3d)
    if conv2d.bias is not None:
        conv3d.bias.data.copy_(conv2d.bias.data)
    return conv3d

def convert_bn2d_to_3d(bn2d):
    bn3d = nn.BatchNorm3d(
        bn2d.num_features, bn2d.eps, bn2d.momentum,
        bn2d.affine, bn2d.track_running_stats
    )
    bn3d.weight.data.copy_(bn2d.weight.data)
    bn3d.bias.data.copy_(bn2d.bias.data)
    bn3d.running_mean.copy_(bn2d.running_mean)
    bn3d.running_var.copy_(bn2d.running_var)
    return bn3d

def convert_pool2d_to_3d(pool2d, pool_type="max"):
    k = pool2d.kernel_size if isinstance(pool2d.kernel_size, tuple) else (pool2d.kernel_size,)*2
    s = pool2d.stride if isinstance(pool2d.stride, tuple) else (pool2d.stride,)*2
    p = pool2d.padding if isinstance(pool2d.padding, tuple) else (pool2d.padding,)*2
    if pool_type == "max":
        return nn.MaxPool3d(kernel_size=(1, *k), stride=(1, *s), padding=(0, *p))
    else:
        return nn.AvgPool3d(kernel_size=(1, *k), stride=(1, *s), padding=(0, *p))

def convert_adaptive_pool2d_to_3d(ap2d):
    osz = ap2d.output_size
    osz = (osz, osz) if isinstance(osz, int) else osz
    return nn.AdaptiveAvgPool3d((1, *osz))

def recursive_inflate(m):
    for name, child in list(m.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(m, name, inflate_conv2d_to_3d(child, DEPTH))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(m, name, convert_bn2d_to_3d(child))
        elif isinstance(child, nn.MaxPool2d):
            setattr(m, name, convert_pool2d_to_3d(child, "max"))
        elif isinstance(child, nn.AvgPool2d):
            setattr(m, name, convert_pool2d_to_3d(child, "avg"))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(m, name, convert_adaptive_pool2d_to_3d(child))
        else:
            recursive_inflate(child)

class DenseNet3D(nn.Module):
    def __init__(self, base2d):
        super().__init__()
        recursive_inflate(base2d)
        old0 = base2d.features.conv0
        out_ch, _, d, h, w = old0.weight.shape
        new0 = nn.Conv3d(1, out_ch, old0.kernel_size, old0.stride, old0.padding, bias=(old0.bias is not None))
        with torch.no_grad():
            new0.weight.copy_(old0.weight.mean(dim=1, keepdim=True))
            if old0.bias is not None:
                new0.bias.copy_(old0.bias)
        base2d.features.conv0 = new0
        self.features = base2d.features
        self.classifier = base2d.classifier

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model():
    base2d = models.densenet121(pretrained=True)
    model = DenseNet3D(base2d).to(DEVICE)
    state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# --- Volume Handling ---
def load_volumes():
    ct = sitk.GetArrayFromImage(sitk.ReadImage(CT_PATH))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(MASK_PATH))
    # Optionally crop to reduce memory usage
    ct = ct[80:144]
    mask = mask[80:144]
    return ct, mask

def split_regions(ct, mask):
    return np.where(mask==1, ct, 0), np.where(mask==2, ct, 0), np.where(mask==0, ct, 0)

def to_tensor(vol):
    return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()

def extract_features(model, region_np):
    t = to_tensor(region_np).to(DEVICE)
    feats = {}
    def hook(name):
        return lambda m, i, o: feats.__setitem__(name, o)
    h2 = model.features.denseblock2.register_forward_hook(hook('b2'))
    h3 = model.features.denseblock3.register_forward_hook(hook('b3'))
    h4 = model.features.denseblock4.register_forward_hook(hook('b4'))
    with torch.no_grad(): _ = model(t)
    h2.remove(); h3.remove(); h4.remove()
    gap = nn.AdaptiveAvgPool3d((1,1,1))
    return {name: gap(fmap).view(-1).cpu().numpy() for name, fmap in feats.items()}

# --- Main Execution ---
def main():
    model = load_model()
    ct, mask = load_volumes()
    tibia, femur, bg = split_regions(ct, mask)

    tibia_feats = extract_features(model, tibia)
    femur_feats = extract_features(model, femur)
    bg_feats    = extract_features(model, bg)

    from scipy.spatial.distance import cosine
    import pandas as pd

    region_names = ['Tibia','Femur','Background']
    region_feats = {'Tibia': tibia_feats,'Femur': femur_feats,'Background': bg_feats}
    layers = ['b2','b3','b4']

    print("Cosine Similarities Between Regions (1.0 = identical, 0.0 = orthogonal):\n")
    results = []
    for layer in layers:
        print(f"→ Layer: {layer}")
        for i in range(len(region_names)):
            for j in range(i+1, len(region_names)):
                r1, r2 = region_names[i], region_names[j]
                v1, v2 = region_feats[r1][layer], region_feats[r2][layer]
                sim = 1 - cosine(v1, v2)
                print(f"  {r1:<9}↔ {r2:<11}: {sim:.4f}")
                results.append({'layer':layer,'pair':f"{r1}-{r2}",'sim':sim})
        print()

    df = pd.DataFrame(results)
    df.to_csv('similarities.csv', index=False)
    print("Similarities saved to similarities.csv")

if __name__ == "__main__":
    main()



# ### The cosine similarity scores quantify how closely the learned feature representations for different anatomical regions align in the 3D DenseNet’s embedding space. 
# 
# #### Tibia ↔ Femur:
# 
# Consistently very high similarity across all layers (0.980 for block2, 0.980 for block3, 0.964 for block4), reflecting that the network’s intermediate and deep features capture common structural patterns shared by both bones.
# 
# #### Tibia ↔ Background & Femur ↔ Background:
# 
# Lower similarity overall, but increasing in deeper layers:
# 
# Block2: Tibia–Background = 0.553, Femur–Background = 0.478
# 
# Block3: Tibia–Background = 0.706, Femur–Background = 0.652
# 
# Block4: Tibia–Background = 0.832, Femur–Background = 0.762
# 
# This trend indicates that early features (block2) strongly differentiate bone from non-bone tissue, while deeper layers (block4) learn more abstract, higher-level representations that partially blur the distinction.
# 
# As a summary:
# 
# The model’s lower-level features (block2) are most discriminative for separating bone vs. background.
# 
# Higher-level features (block4) emphasize shared semantic information—resulting in increased similarity between bone and background.
# 
# The very high tibia–femur similarity at every depth confirms that the network encodes common bone characteristics, which is desirable given the anatomical and textural resemblance of these regions.
