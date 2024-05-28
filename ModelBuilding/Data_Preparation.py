#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
import os
import random
import csv


# In[2]:


# loading dataset
datafolder = r'./EUVP/Paired'


# In[3]:


underwater_img_paths = []
groundtruth_img_paths = []

for folder_name in os.listdir(datafolder):
  underwater_img_dir = os.path.join(datafolder,folder_name,'trainA')
  groundtruth_img_dir = os.path.join(datafolder,folder_name,'trainB')

  # Check folders are exists
  print(f'============== {folder_name} ==============')
  if not os.path.exists(underwater_img_dir):
    print(f"{underwater_img_dir} : FOLDER NOT FOUND")
    continue

  if not os.path.exists(groundtruth_img_dir):
    print(f"{groundtruth_img_dir} : FOLDER NOT FOUND")
    continue

  underwater_img_names = os.listdir(underwater_img_dir)
  for underwater_img_name in underwater_img_names:
    underwater_img_path = os.path.join(underwater_img_dir,underwater_img_name)
    groundtruth_img_path = os.path.join(groundtruth_img_dir,underwater_img_name)

    # check groundtruth image exists
    if not os.path.exists(groundtruth_img_path):
      print(f"{groundtruth_img_path} : FOLDER NOT FOUND")
      continue

    underwater_img_paths.append(underwater_img_path)
    groundtruth_img_paths.append(groundtruth_img_path)

  print(f'n_underwater_images : {len(underwater_img_paths)} | n_groundtruth_images : {len(groundtruth_img_paths)} \n')


# In[4]:


# Soritng images
underwater_img_paths = sorted(underwater_img_paths)
groundtruth_img_paths = sorted(groundtruth_img_paths)


# In[5]:


#Train

TRAIN_SPLIT      = 0.7
VALIDATION_SPLIT = 0.1
TEST_SPLIT       = 0.2


# In[6]:


# Combine the groundtruth and noise image paths into tuples
image_pairs = list(zip(underwater_img_paths,groundtruth_img_paths))

# Shuffle the image pairs randomly
random.shuffle(image_pairs)

# Calculate the number of samples for each split based on the ratios
total_samples = len(image_pairs)
train_samples = int(TRAIN_SPLIT * total_samples)
val_samples = int(VALIDATION_SPLIT * total_samples)
test_samples = total_samples - train_samples - val_samples

# Split the data into train, validation, and test sets
train_data = image_pairs[:train_samples]
val_data = image_pairs[train_samples:train_samples + val_samples]
test_data = image_pairs[train_samples + val_samples:]


# In[7]:


### Optional
# Save the data into separate CSV files
def save_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['UnderWater Images', 'GroundTruth Images'])
        writer.writerows(data)

save_to_csv('train.csv', train_data)
save_to_csv('validation.csv', val_data)
save_to_csv('test.csv', test_data)


# In[ ]:




