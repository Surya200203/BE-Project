#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb

# Define enhancement functions
def cal_equalisation(img, ratio):
    Array = img * ratio
    Array = np.clip(Array, 0, 255)
    return Array

def RGB_equalisation(img, rgb_scale_range):
    img = np.float32(img)
    avg_RGB = []
    for i in range(3):
        avg = np.mean(img[:, :, i])
        avg_RGB.append(avg)
    
    # Calculate the scaling factors for each channel
    a_r = avg_RGB[0] / avg_RGB[2]
    a_g = avg_RGB[0] / avg_RGB[1]
    ratio = [1, a_g, a_r]
    
    # Apply a more conservative enhancement
    for i in range(1, 3):
        # Ensure that the scaling factor does not deviate too much from 1
        min_scale, max_scale = rgb_scale_range
        if ratio[i] > max_scale:
            ratio[i] = max_scale
        elif ratio[i] < min_scale:
            ratio[i] = min_scale
        img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])
    
    return img

def histogram_stretch(channel, height, width, percentile):
    length = height * width
    sorted_values = sorted(channel.flatten())
    I_min = int(sorted_values[int(length * percentile)])
    I_max = int(sorted_values[-int(length * percentile)])
    
    stretched_channel = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if channel[i][j] < I_min:
                stretched_channel[i][j] = I_min
            elif channel[i][j] > I_max:
                stretched_channel[i][j] = 255
            else:
                stretched_channel[i][j] = int((channel[i][j] - I_min) * (255 / (I_max - I_min)))
    
    return stretched_channel

def stretching(img, percentile):
    height = len(img)
    width = len(img[0])
    img[:, :, 2] = histogram_stretch(img[:, :, 2], height, width, percentile)
    img[:, :, 1] = histogram_stretch(img[:, :, 1], height, width, percentile)
    img[:, :, 0] = histogram_stretch(img[:, :, 0], height, width, percentile)
    return img

def HSVStretching(sceneRadiance, stretch_factor):
    sceneRadiance = np.uint8(sceneRadiance)
    height, width, _ = sceneRadiance.shape
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = global_stretching(s, height, width, stretch_factor)
    img_v_stretching = global_stretching(v, height, width, stretch_factor)

    # Preserve original hue values
    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255

    return img_rgb

def global_stretching(img_L, height, width, stretch_factor):
    I_min = np.percentile(img_L, 100 * stretch_factor)
    I_max = np.percentile(img_L, 100 * (1 - stretch_factor))

    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            p_out = (img_L[i][j] - I_min) * (1 / (I_max - I_min))
            array_Global_histogram_stretching_L[i][j] = p_out

    return array_Global_histogram_stretching_L

def sceneRadianceRGB(sceneRadiance):
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def enhance_image(img, rgb_scale_range, hist_percentile, hsv_stretch_factor):
    sceneRadiance = RGB_equalisation(img, rgb_scale_range)
    sceneRadiance = stretching(sceneRadiance, hist_percentile)
    sceneRadiance = HSVStretching(sceneRadiance, hsv_stretch_factor)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    return sceneRadiance

def process_and_enhance_image(input_image, param_combinations=[ ((0.995, 1.005), 0.001, 0.002)]):
    # Read the images
    # input_img = cv2.imread(input_image_path)
    # ground_truth_img = cv2.imread(ground_truth_image_path)

    # Check if the images were successfully read
    if input_image is not None:
        for i, params in enumerate(param_combinations):
            print(f'Testing parameters set {i + 1}: {params}')
            rgb_scale_range, hist_percentile, hsv_stretch_factor = params

            # Apply image enhancement techniques to the input image
            enhanced_img = enhance_image(input_image, rgb_scale_range, hist_percentile, hsv_stretch_factor)
        
            
            return enhanced_img  # Return the enhanced image
        
    else:
        if input_img is None:
            print(f'Error: Unable to read the input image file {input_image_path}')
        # if ground_truth_img is None:
        #     print(f'Error: Unable to read the ground truth image file {ground_truth_image_path}')

# Example usage
# input_image_path = '../OUTPUT/test_p116_.jpg'
# ground_truth_image_path = '../EUVP/EUVP/test_samples/GTr/test_p116_.jpg'

# enhanced_image = process_and_enhance_image(input_image_path, ground_truth_image_path)

