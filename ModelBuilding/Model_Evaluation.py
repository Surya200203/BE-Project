#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
from UCM import process_and_enhance_image


# In[3]:


def psnr_loss_fn(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=1.0)

def ssim_loss_fn(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)


# In[4]:


model = load_model('./uidudl_beproject.h5', custom_objects={'psnr_loss_fn': psnr_loss_fn, 'ssim_loss_fn': ssim_loss_fn})


# In[5]:


tf.config.run_functions_eagerly(True)


# In[6]:


# Load train test validation image paths form csv files 
def load_excel_data(file_path, column1_name='UnderWater Images', column2_name='GroundTruth Images', sheet_name="Sheet1"):
    try:
        # Load the Excel file
        df = pd.read_csv(file_path)

        # Extract the data from the specified columns
        column1_data = df[column1_name].tolist()
        column2_data = df[column2_name].tolist()

        return column1_data, column2_data

    except Exception as e:
        print(f"Error occurred while loading data from Excel: {e}")
        return None, None


# In[7]:


test_csv_path = r'./test.csv'
test_x_paths, test_y_paths = load_excel_data(test_csv_path)


# In[8]:


test_x_paths = sorted(test_x_paths)
test_y_paths = sorted(test_y_paths)


# In[9]:


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

# Remove Padding
# def inverse_padding(pad_image,image_dim,pad_method="center_padding"):
#   pad_img_height = pad_image.shape[0]
#   pad_img_width = pad_image.shape[1]

#   img_height = image_dim[0]
#   img_width = image_dim[1]

#   if pad_method == "center_padding":
#     pad_y1 = (pad_img_height - img_height)//2
#     if pad_y1*2 == (pad_img_height - img_height):pad_y2 = pad_y1
#     else: pad_y2 = pad_y1+1

#     pad_x1 = (pad_img_width - img_width)//2
#     if pad_x1*2 == (pad_img_width - img_width):pad_x2 = pad_x1
#     else: pad_x2 = pad_x1+1
#     extract_image = pad_image[pad_y1:pad_img_height-pad_y2,pad_x1:pad_img_width-pad_x2]

#   if pad_method == "corner_padding":
#     extract_image = pad_image[0:img_height,0:img_width]


#   return extract_image


# In[10]:


@tf.function
def load_image_file(file_path):
    file_path = file_path.numpy().decode("utf-8")
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(256,256))
    # img = cv2.resize(img,(0,0),fx=0.5, fy=0.5)
    preprocess_img = pad_image(img)
    preprocess_img = preprocess_img/255
    return preprocess_img

def image_dataset(image_list):
    files = tf.data.Dataset.from_tensor_slices(image_list)
    dataset = files.map(lambda x: tf.py_function(load_image_file, [x], tf.float32))
    return dataset


# In[11]:


BATCH_SIZE = 8


# In[12]:


test_x = image_dataset(list(test_x_paths))
test_y = image_dataset(list(test_y_paths))

# combine input and output
test = tf.data.Dataset.zip((test_x, test_y))
# test = test.shuffle(100)
test = test.batch(BATCH_SIZE)
# test.prefetch(tf.data.AUTOTUNE)


# In[13]:


test


# In[14]:


# def evaluate_model(model, test_dataset):
#     psnr_values = []
#     ssim_values = []

#     for x_img, y_img in test_dataset:
#         # Predict
#         prediction = model.predict(x_img)
#         print(prediction.dtype)
#         postprocessed_img = process_and_enhance_image(prediction)
        

#         # Calculate PSNR and SSIM
#         psnr = psnr_loss_fn(postprocessed_img, y_img)
#         ssim = ssim_loss_fn(postprocessed_img, y_img)

#         psnr_values.append(psnr.numpy())
#         ssim_values.append(ssim.numpy())

#     # Compute average PSNR and SSIM
#     avg_psnr = np.mean(psnr_values)
#     avg_ssim = np.mean(ssim_values)

#     return avg_psnr, avg_ssim


# In[15]:


import numpy as np

def evaluate_model(model, test_dataset):
    psnr_values = []
    ssim_values = []

    for x_img, y_img in test_dataset:
        # Predict
        prediction = model.predict(x_img)
        output_img = prediction * 255.0
        output_img = np.clip(output_img, 0, 255).astype(np.uint8) 
        print(prediction.dtype)
        postprocessed_img = process_and_enhance_image(output_img)
        

        # Calculate PSNR and SSIM
        psnr = psnr_loss_fn(postprocessed_img, y_img)
        ssim = ssim_loss_fn(postprocessed_img, y_img)

        psnr_values.append(psnr.numpy())
        ssim_values.append(ssim.numpy())

    # Compute average PSNR and SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim


# In[16]:


# Evaluate the model
avg_psnr, avg_ssim = evaluate_model(model, test)

print("Average PSNR:", avg_psnr)
print("Average SSIM:", avg_ssim)


# In[ ]:





# In[ ]:


import os
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tensorflow.keras.models import load_model
from UCM import process_and_enhance_image

# Define paths to your input and ground truth folders
input_folder = './EUVP/test_samples/Inp'
ground_truth_folder = './EUVP/test_samples/GTr'

# Load your trained model
model = load_model('./uidudl_beproject.h5', custom_objects={'psnr_loss_fn': psnr_loss_fn, 'ssim_loss_fn': ssim_loss_fn})
# model = load_model('./uidudl_beproject.h5')


# Function to load images
def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)
    filenames.sort()  # Make sure the order of images is the same for input and ground truth
    for filename in filenames:
        img = io.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img_as_float(img))  # Convert images to float in range [0, 1]
    return images, filenames

# Load input and ground truth images
input_images, input_filenames = load_images_from_folder(input_folder)
ground_truth_images, ground_truth_filenames = load_images_from_folder(ground_truth_folder)

# Ensure both folders contain the same number of images
assert len(input_images) == len(ground_truth_images), "Number of images in input and ground truth folders must match"

psnr_values = []
ssim_values = []

# Loop through images and calculate PSNR and SSIM
for i in range(len(input_images)):
    input_image = np.expand_dims(input_images[i], axis=0)  # Add batch dimension
    enhanced_image = model.predict(input_image)[0]  # Get the enhanced image
    post_processed_image = process_and_enhance_image(enhanced_image)
    ground_truth_image = ground_truth_images[i]

    # Ensure the shapes match
    assert post_processed_image.shape == ground_truth_image.shape, "Shape of enhanced image and ground truth image must match"

    # Calculate PSNR
    psnr_value = psnr(ground_truth_image, post_processed_image, data_range=post_processed_image.max() - post_processed_image.min())

    # Set win_size based on the smallest dimension of the image
    min_dimension = min(ground_truth_image.shape[:2])
    win_size = min(min_dimension, 7) if min_dimension % 2 == 1 else min(min_dimension - 1, 7)

    # Set channel_axis if the image is multichannel
    if ground_truth_image.ndim == 3 and ground_truth_image.shape[2] == 3:
        ssim_value, _ = ssim(ground_truth_image, post_processed_image, full=True, data_range=post_processed_image.max() - post_processed_image.min(), win_size=win_size, channel_axis=2)
    else:
        ssim_value, _ = ssim(ground_truth_image, post_processed_image, full=True, data_range=post_processed_image.max() - post_processed_image.min(), win_size=win_size)

    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    print(f"Image: {input_filenames[i]}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")

# Calculate average PSNR and SSIM
average_psnr = np.mean(psnr_values)
average_ssim = np.mean(ssim_values)

print(f"Average PSNR: {average_psnr:.4f}")
print(f"Average SSIM: {average_ssim:.4f}")


# In[3]:


import os
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tensorflow.keras.models import load_model
from UCM import process_and_enhance_image
import matplotlib.pyplot as plt

# Define paths to your input and ground truth folders
input_folder = './EUVP/test_samples/Inp'
ground_truth_folder = './EUVP/test_samples/GTr'

# Load your trained model
model = load_model('./uidudl_beproject.h5', custom_objects={'psnr_loss_fn': psnr_loss_fn, 'ssim_loss_fn': ssim_loss_fn})

# Function to load images
def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)
    filenames.sort()  # Make sure the order of images is the same for input and ground truth
    for filename in filenames:
        img = io.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img_as_float(img))  # Convert images to float in range [0, 1]
    return images, filenames

# Load input and ground truth images
input_images, input_filenames = load_images_from_folder(input_folder)
ground_truth_images, ground_truth_filenames = load_images_from_folder(ground_truth_folder)

# Ensure both folders contain the same number of images
assert len(input_images) == len(ground_truth_images), "Number of images in input and ground truth folders must match"

psnr_values = []
ssim_values = []

# Loop through images and calculate PSNR and SSIM
for i in range(len(input_images)):
    input_image = np.expand_dims(input_images[i], axis=0)  # Add batch dimension
    enhanced_image = model.predict(input_image)[0]  # Get the enhanced image
    post_processed_image = process_and_enhance_image(enhanced_image)
    ground_truth_image = ground_truth_images[i]

    # Ensure the shapes match
    assert post_processed_image.shape == ground_truth_image.shape, "Shape of enhanced image and ground truth image must match"

    # Calculate PSNR
    psnr_value = psnr(ground_truth_image, post_processed_image, data_range=post_processed_image.max() - post_processed_image.min())

    # Set win_size based on the smallest dimension of the image
    min_dimension = min(ground_truth_image.shape[:2])
    win_size = min(min_dimension, 7) if min_dimension % 2 == 1 else min(min_dimension - 1, 7)

    # Set channel_axis if the image is multichannel
    if ground_truth_image.ndim == 3 and ground_truth_image.shape[2] == 3:
        ssim_value, _ = ssim(ground_truth_image, post_processed_image, full=True, data_range=post_processed_image.max() - post_processed_image.min(), win_size=win_size, channel_axis=2)
    else:
        ssim_value, _ = ssim(ground_truth_image, post_processed_image, full=True, data_range=post_processed_image.max() - post_processed_image.min(), win_size=win_size)

    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_images[i])
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[1].imshow(post_processed_image)
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')
    axes[2].imshow(ground_truth_image)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    plt.show()

    print(f"Image: {input_filenames[i]}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")

# Calculate average PSNR and SSIM
average_psnr = np.mean(psnr_values)
average_ssim = np.mean(ssim_values)

print(f"Average PSNR: {average_psnr:.4f}")
print(f"Average SSIM: {average_ssim:.4f}")


# In[6]:


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io

# Load the images
image1 = io.imread('./post_processd.jpg')
image2 = io.imread(r"C:\Users\asp40\OneDrive\Desktop\BEPROJECT\EUVP\EUVP\test_samples\GTr\test_p0_.jpg")

# Calculate PSNR
psnr_value = psnr(image1, image2, data_range=image2.max() - image2.min())

print(f"PSNR between the images: {psnr_value:.2f} dB")


# In[5]:


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io, transform

# Load the images
image1 = io.imread('./post_processd.jpg')
image2 = io.imread(r"C:\Users\asp40\OneDrive\Desktop\BEPROJECT\EUVP\EUVP\test_samples\GTr\test_p0_.jpg")

# Resize images if they are smaller than the minimum required size for SSIM calculation
min_size = 7  # Minimum size required for SSIM calculation
if min(image1.shape) < min_size or min(image2.shape) < min_size:
    scale_factor = min_size / min(min(image1.shape), min(image2.shape))
    image1 = transform.rescale(image1, scale_factor, anti_aliasing=True)
    image2 = transform.rescale(image2, scale_factor, anti_aliasing=True)

# Calculate PSNR
psnr_value = psnr(image1, image2, data_range=image2.max() - image2.min())

# Calculate SSIM
ssim_value = ssim(image1, image2, multichannel=True, data_range=image2.max() - image2.min())

print(f"PSNR between the images: {psnr_value:.2f} dB")
print(f"SSIM between the images: {ssim_value:.2f}")






import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io, transform

# Path to the folders containing the images
output_folder = r"C:\Users\asp40\OneDrive\Desktop\BEPROJECT\EUVP\EUVP\test_samples\PostProcessed"
ground_truth_folder =r"C:\Users\asp40\OneDrive\Desktop\BEPROJECT\EUVP\EUVP\test_samples\GTr"

# List all files in the output folder
output_files = os.listdir(output_folder)

# Iterate over each file in the output folder
for file_name in output_files:
    # Check if the file exists in the ground truth folder
    if file_name in os.listdir(ground_truth_folder):
        # Load the images
        output_image = io.imread(os.path.join(output_folder, file_name))
        ground_truth_image = io.imread(os.path.join(ground_truth_folder, file_name))
        
        # Resize images if necessary
        min_size = 7  # Minimum size required for SSIM calculation
        if min(output_image.shape) < min_size or min(ground_truth_image.shape) < min_size:
            scale_factor = min_size / min(min(output_image.shape), min(ground_truth_image.shape))
            output_image = transform.rescale(output_image, scale_factor, anti_aliasing=True)
            ground_truth_image = transform.rescale(ground_truth_image, scale_factor, anti_aliasing=True)
        
        # Calculate PSNR
        psnr_value = psnr(output_image, ground_truth_image, data_range=ground_truth_image.max() - ground_truth_image.min())
        
        # Calculate SSIM
        ssim_value = ssim(output_image, ground_truth_image, multichannel=True, data_range=ground_truth_image.max() - ground_truth_image.min())
        
        print(f"Image: {file_name}")
        print(f"PSNR between the images: {psnr_value:.2f} dB")
        print(f"SSIM between the images: {ssim_value:.2f}")
        print("\n")
    else:
        print(f"Corresponding image not found for {file_name} in ground truth folder.")




import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io, transform

# Path to the folders containing the images
output_folder = r"C:\Users\asp40\OneDrive\Desktop\BEPROJECT\EUVP\EUVP\test_samples\PostProcessed"
ground_truth_folder =r"C:\Users\asp40\OneDrive\Desktop\BEPROJECT\EUVP\EUVP\test_samples\GTr"

# Initialize variables to accumulate PSNR and SSIM values
total_psnr = 0
total_ssim = 0
num_images = 0

# List all files in the output folder
output_files = os.listdir(output_folder)

# Iterate over each file in the output folder
for file_name in output_files:
    # Check if the file exists in the ground truth folder
    if file_name in os.listdir(ground_truth_folder):
        # Load the images
        output_image = io.imread(os.path.join(output_folder, file_name))
        ground_truth_image = io.imread(os.path.join(ground_truth_folder, file_name))
        
        # Resize images if necessary
        min_size = 7  # Minimum size required for SSIM calculation
        if min(output_image.shape) < min_size or min(ground_truth_image.shape) < min_size:
            scale_factor = min_size / min(min(output_image.shape), min(ground_truth_image.shape))
            output_image = transform.rescale(output_image, scale_factor, anti_aliasing=True)
            ground_truth_image = transform.rescale(ground_truth_image, scale_factor, anti_aliasing=True)
        
        # Calculate PSNR
        psnr_value = psnr(output_image, ground_truth_image, data_range=ground_truth_image.max() - ground_truth_image.min())
        
        # Calculate SSIM
        ssim_value = ssim(output_image, ground_truth_image, multichannel=True, data_range=ground_truth_image.max() - ground_truth_image.min())
        
        print(f"Image: {file_name}")
        print(f"PSNR between the images: {psnr_value:.2f} dB")
        print(f"SSIM between the images: {ssim_value:.2f}")
        print("\n")
        
        # Accumulate PSNR and SSIM values
        total_psnr += psnr_value
        total_ssim += ssim_value
        num_images += 1
        
    else:
        print(f"Corresponding image not found for {file_name} in ground truth folder.")

# Calculate average PSNR and SSIM values
average_psnr = total_psnr / num_images
average_ssim = total_ssim / num_images

# Print average PSNR and SSIM values
print(f"Average PSNR across all images: {average_psnr:.2f} dB")
print(f"Average SSIM across all images: {average_ssim:.2f}")


# In[ ]:




