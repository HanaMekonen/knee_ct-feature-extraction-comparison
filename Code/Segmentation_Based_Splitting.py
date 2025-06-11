#!/usr/bin/env python
# coding: utf-8

# ## 1. Segmentation-Based Splitting:

# On this step, rather than performing bone segmentation from scratch, I leveraged the pre‐segmented mask generated in Task I (repository name: knee-tibia-femur-segmentation ). This mask—already delineating the tibia, femur, and non‐bone regions, was loaded directly and used to isolate each anatomical structure.

# In[1]:


# Load the CT scan and segmentation mask using SimpleITK:

import SimpleITK as sitk
import numpy as np

# Load the 3D volume (both CT and segmented mask assumed to be in one file)
seg_image = sitk.ReadImage('bone_segmented.nii.gz')
seg_array = sitk.GetArrayFromImage(seg_image)  # Shape: [slices, height, width]


# In[2]:


print("Labels in segmentation:", np.unique(seg_array))


# In[3]:


# Separate the regions by masking out unwanted voxels:

# Create individual region masks
tibia_region = np.where(seg_array == 1, 1.0, 0.0)        # Tibia
femur_region = np.where(seg_array == 2, 1.0, 0.0)        # Femur
background_region = np.where(seg_array == 0, 1.0, 0.0)   # Background / non-bone


# In[4]:


# multiply these masks with the original CT array, in order to apply the model to real intensity values

ct_image = sitk.ReadImage('3702_left_knee.nii.gz')
ct_array = sitk.GetArrayFromImage(ct_image)

tibia_ct = ct_array * tibia_region
femur_ct = ct_array * femur_region
background_ct = ct_array * background_region

# Each of tibia_ct, femur_ct, and background_ct will contain only the intensities from the respective region; all other voxels will be zero


# In[7]:


print("Segmentation shape:", seg_array.shape)
print("Tibia values:", np.unique(tibia_region))
print("Femur values:", np.unique(femur_region))
print("Background values:", np.unique(background_region))


# In[8]:


print("Tibia voxels:", np.sum(tibia_region))
print("Femur voxels:", np.sum(femur_region))
print("Background voxels:", np.sum(background_region))


# Voxel Count Analysis
# Tibia voxels: 253,582
# → we have ~250K voxels labeled as tibia. This indicates a substantial and valid region in our mask.
# 
# Femur voxels: 203,879
# → Likewise, the femur region is well represented.
# 
# Background voxels: 56,165,643
# → This large number is expected since background typically fills most of the CT volume.

# In[9]:


# Convert back to SimpleITK image and save
tibia_img = sitk.GetImageFromArray(tibia_ct)
tibia_img.CopyInformation(ct_image)  # preserve spacing/origin/direction
sitk.WriteImage(tibia_img, 'tibia_region.nii.gz')


femur_img = sitk.GetImageFromArray(femur_ct)
femur_img.CopyInformation(ct_image)  
sitk.WriteImage(femur_img, 'femur_region.nii.gz')

# Convert back to SimpleITK image and save
background_img = sitk.GetImageFromArray(background_ct)
background_img.CopyInformation(ct_image) 
sitk.WriteImage(background_img, 'background.nii.gz')


