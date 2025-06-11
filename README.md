# 3D Knee CT Feature Extraction & Comparison Pipeline

## Overview  
This repository implements a complete pipeline to extract and compare deep‐learning–based features from segmented regions of a 3D knee CT. 
We leverage a pretrained 2D DenseNet121, inflate it to 3D, and compute cosine similarities between Tibia, Femur, and Background regions at multiple network depths.

## Repository Structure  

```
knee-feature-extraction/
├── code/
│   ├── Segmentation_Based_Splitting.py               # Functions to split CT into tibia/femur/background
│   ├── build_model3d.py                              # Convert 2D DenseNet to 3D
│   ├── Feature_Extraction&Feature_Comparison.py      # loads volumes, splits regions, extracts feature vectors, computes & saves cosine similarities
├── results/
│   ├── features/                     # Numpy feature vectors
│   ├── similarities.csv              # Output cosine similarity scores
├── data/
│   ├── 3702_left_knee.nii.gz         # Sample CT volume
│   └── bone_segmented.nii.gz         # Labeled mask for segmentation
├── requirements.txt
├── .gitignore
└── README.md
```


## Requirements
```
- Python 3.7+  
- PyTorch 1.7+  
- torchvision  
- SimpleITK  
- NumPy  
- SciPy  

Install dependencies:

pip install -r requirements.txt
```
### Data
knee_ct.nii.gz – 3D knee CT volume.
bone_segmented.nii.gz – 3D label mask from Task I, encoding Tibia=1, Femur=2, Background=0.

## Pipeline Steps

### Step 1: Segmentation-Based Splitting

python Segmentation_Based_Splitting.py

This step isolates the three anatomical regions (Tibia, Femur, Background) from the provided segmentation mask.

### Step 2: Build & Save 3D Model

python build_model3d.py

Loads pretrained 2D DenseNet121

Recursively inflates Conv2d→Conv3d, BatchNorm2d→BatchNorm3d, pooling layers

Collapses first conv to accept single-channel input

Wraps forward to use 3D pooling

Saves model3d_state_dict.pth

### Step 3: Feature Extraction and Feature Comparison

python Feature_Extraction&Feature_Comparison.py

Loads model3d_state_dict.pth on CPU

Reads knee_ct.nii.gz and bone_segmented.nii.gz with SimpleITK

Splits into three volumes (Tibia, Femur, Background) by mask labels

Runs each region through the 3D model in eval mode

Hooks outputs of denseblock2, denseblock3, denseblock4

Applies 3D global average pooling → fixed‐length vectors

Saves all vectors in features_cpu.npz

Computes cosine similarity for each region pair (Tibia↔Femur, Tibia↔Background, Femur↔Background)

Performs comparison at each layer (b2, b3, b4)

Prints results and writes similarities.csv

## 📦 Outputs

features_cpu.npz - Extracted features

similarities.csv - cosine similarity scores for all region pairs (Tibia vs Femur, Tibia vs Background, etc.) across 3 layers


## 🧠 References

PyTorch DenseNet

SimpleITK Documentation
