#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import random


import tensorflow as tf
# from tensorflow.keras.layers import Dense, Activation, Concatenate, GlobalAveragePooling2D, Multiply,GlobalMaxPooling2D
# from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow.keras.layers import MaxPool2D,Conv2D,UpSampling2D,Input,Dropout,Activation
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from keras import layers
import keras



tf.config.run_functions_eagerly(True)



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





train_csv_path = r'./train.csv'
test_csv_path = r'./test.csv'
validation_csv_path = r'./validation.csv'


train_x_paths, train_y_paths = load_excel_data(train_csv_path)
val_x_paths, val_y_paths = load_excel_data(validation_csv_path)
test_x_paths, test_y_paths = load_excel_data(test_csv_path)





train_x_paths = sorted(train_x_paths)
train_y_paths = sorted(train_y_paths)
val_x_paths = sorted(val_x_paths)
val_y_paths = sorted(val_y_paths)
test_x_paths = sorted(test_x_paths)
test_y_paths = sorted(test_y_paths)




print(f'X_train : {len(train_x_paths)}')
print(f'Y_train : {len(train_y_paths)}')
print(f'X_val   : {len(val_x_paths)}')
print(f'Y_val   : {len(val_y_paths)}')
print(f'X_test  : {len(test_x_paths)}')
print(f'Y_test  : {len(test_y_paths)}')





gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)





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




BATCH_SIZE = 8




train_x = image_dataset(list(train_x_paths))
train_y = image_dataset(list(train_y_paths))


# combine input and output
train = tf.data.Dataset.zip((train_x, train_y))
# train = train.take(100)
# train = train.shuffle(100)
train = train.batch(BATCH_SIZE)
train.prefetch(tf.data.AUTOTUNE)




val_x = image_dataset(list(val_x_paths))
val_y = image_dataset(list(val_y_paths))

# combine input and output
val = tf.data.Dataset.zip((val_x, val_y))
# train = train.shuffle(100)
val = val.batch(BATCH_SIZE)
val.prefetch(tf.data.AUTOTUNE)



test_x = image_dataset(list(test_x_paths))
test_y = image_dataset(list(test_y_paths))

# combine input and output
test = tf.data.Dataset.zip((test_x, test_y))
# test = test.shuffle(100)
test = test.batch(BATCH_SIZE)
test.prefetch(tf.data.AUTOTUNE)




def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    downsample.add(Dropout(0.25))
    return downsample


def up(filters, kernel_size, dropout = True):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    upsample.add(Activation("sigmoid"))
    return upsample




def build_model():
    inputs = layers.Input(shape= [None,None,3])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(128,(3,3),False)(d1)
    d3 = down(256,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)   
    d5 = down(512,(3,3),True)(d4)
    d6 = down(1024,(3,3),True)(d5)
    #upsampling
    u0 = up(512,(3,3),False)(d6)
    u0 = layers.concatenate([u0,d5])
    u1 = up(512,(3,3),False)(u0)
    u1 = layers.concatenate([u1,d4])
    u2 = up(256,(3,3),False)(u1)
    u2 = layers.concatenate([u2,d3])
    u3 = up(128,(3,3),False)(u2)
    u3 = layers.concatenate([u3,d2])
    u4 = up(128,(3,3),False)(u3)
    u4 = layers.concatenate([u4,d1])
    u5 = up(3,(3,3),False)(u4)
    u5 = layers.concatenate([u5,inputs])
    output = layers.Conv2D(3,(2,2),strides = 1, padding = 'same')(u5)
#     output = layers.Dense(1, activation='sigmoid')(output1)
    return tf.keras.Model(inputs=inputs, outputs=output)




model = build_model()
model.summary()




# print("trainable_weights:", len(model.trainable_weights))
# print("non_trainable_weights:", len(model.non_trainable_weights))


# In[18]:


def psnr_loss_fn(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=1.0)

def ssim_loss_fn(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)


# In[20]:


from keras.callbacks import ModelCheckpoint

# tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


# In[21]:


# from datetime import datetime
# model_save_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_save_dir = f'models/{model_save_folder}'
# #model_save_dir = f'models'

# if not os.path.exists(model_save_dir):
#   print("FOLDER CREATED")
#   os.makedirs(model_save_dir)

# save_best_model_checkpoint = ModelCheckpoint(model_save_dir+'/model-{epoch:03d}.hdf5',monitor='val_loss',save_best_only=True,mode='auto')


# In[22]:


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_squared_error', metrics = [psnr_loss_fn,ssim_loss_fn])


# In[23]:


hist = model.fit(train, epochs = 20, validation_data = val, callbacks=[tensorboard_callback])


# In[25]:


# save weights
model.save("uidudl_beproject.h5")
# tf.keras.models.save_model("./uidudl_beproject.h5")

# save entire model
model.save('uidudl_beproject')

model.save('uidudl_beproject.keras')


# In[26]:


plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val loss')
plt.suptitle('Loss')
plt.legend()
plt.show()


# In[27]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[28]:


get_ipython().run_line_magic('tensorboard', "--logdir './logs'")


# In[ ]:




