import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt

def psnr_loss_fn(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=1.0)

def ssim_loss_fn(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)

model = load_model('./uidudl_beproject.h5', custom_objects={'psnr_loss_fn': psnr_loss_fn, 'ssim_loss_fn': ssim_loss_fn})

# tf.config.run_functions_eagerly(True)

def padding_calc(input_dim,multiplier=32):
    return math.ceil(input_dim/multiplier)*multiplier - input_dim

# Add Padding
def pad_image(image,mood = "center_padding"):
    img_h = image.shape[0]
    img_w = image.shape[1]

    pad_y = padding_calc(img_h)
    pad_x = padding_calc(img_w)

    if mood == "center_padding":
        pad_y2 = pad_y//2
        pad_x2 = pad_x//2

        padded_img = image.copy()
        if pad_y%2 != 0:
            padded_img = np.pad(image, ((pad_y2, pad_y2+1), (pad_x2, pad_x2), (0, 0)), mode='constant')
        if pad_x%2 != 0:
            padded_img = np.pad(image, ((pad_y2, pad_y2), (pad_x2, pad_x2+1), (0, 0)), mode='constant')
        if (pad_y%2 == 0) & (pad_x%2 == 0):
            padded_img = np.pad(image, ((pad_y2, pad_y2), (pad_x2, pad_x2), (0, 0)), mode='constant')

    elif mood == "corner_padding":
        padded_img = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
    return padded_img

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(256,256))
    preprocess_img = pad_image(img)
    preprocess_img = preprocess_img/255;
    return preprocess_img

import cv2
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from glob import glob
from PostProcessing import process_and_enhance_image
import os



def read_single_image(input_image,image_file_name, output_folder='./output'):
    if not input_image :
        print("No images found. Please provide valid image paths.")
        return

    input_img = preprocess_image(input_image)
    
    
    input_img = np.expand_dims(input_img, axis=0)
    output_img = model.predict(input_img)[0]
    output_img = output_img * 255.0
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
             
    postprocessed_img = process_and_enhance_image(output_img)
    
    # Extract filename from input path
    # filename = os.path.basename(image_file_name)
    output_path = os.path.join(output_folder, image_file_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    cv2.imwrite(output_path, cv2.cvtColor(postprocessed_img, cv2.COLOR_BGR2RGB))
    print(f'Saved post-processed image to {output_path}')

# # Example usage
# # input_image_path = r"C:\Users\asp40\OneDrive\Desktop\BEPROJECT\EUVP\EUVP\test_samples\Inp\test_p110_.jpg"

# # read_single_image(input_image_path)