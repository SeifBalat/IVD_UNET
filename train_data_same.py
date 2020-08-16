import os.path
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.image as img
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from itertools import chain
from skimage import color
import skimage.filters
import skimage.viewer
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from natsort import natsorted, ns
from UNet import get_unet
import glob

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

seed = 42
np.random.seed = seed
np.random.seed = seed

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 64
IMG_CHANNELS = 3

TRAIN_PATH = "B:/Masterarbeit/Thesis/Dataset/IVDM_Update/train/"
TEST_PATH = "B:/Masterarbeit/Thesis/Dataset/IVDM_Update/test/"
mask_Path = "B:/Masterarbeit/Thesis/Dataset/IVDM_Update/train/"

#walk through the folders(train images, train masks & test images )
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
mask_ids = next(os.walk(mask_Path))[1]

#Define three zeros arrays (train images, train masks & test images )
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
#type of boolen if True or False(mask)
Y_train = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Resizing test images') 

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    test_path = TEST_PATH + id_ + '/Fat/' +'Axial/' 
    test_images = next(os.walk(test_path))[2]
    
    #how we sort the images by default 
    #sort a list of strings that contain numbers, the normal python sort algorithm sorts lexicographically"
    test_images = natsorted(test_images, alg=ns.IGNORECASE)
    #create zeros array to collect the cutting images. 
    collective_test_images = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    #walk through the folder and read the test images 
    for m, test_image_name in tqdm(enumerate(test_images), total=len(test_images)):
        #read the image
        test_img = imread(test_path + test_image_name)[:,:,:IMG_CHANNELS]
        #resize the images with new height and new width.
        test_img = resize(test_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        #collecting the images to use them later on.
        collective_test_images = np.maximum(collective_test_images, test_img)
    X_test[n] = collective_test_images

print(type(collective_test_images))
print(collective_test_images.shape)

#we are doing the same with the train images and it's mask
print('Resizing training images and masks')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): 
    
    train_axial_path_images = TRAIN_PATH + id_ + '/Fat/' +'Axial/'
    train_axial_path_masks = TRAIN_PATH + id_ + '/GT/' +'Axial/'
    #train_images is a list 
    train_images = next(os.walk(train_axial_path_images))[2]
    train_masks = next(os.walk(train_axial_path_masks))[2]
    #how we sort the images folder 
    train_images= natsorted(train_images, alg=ns.IGNORECASE)
    train_masks= natsorted(train_masks, alg=ns.IGNORECASE)
    collective_images = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    collective_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)

    for m, image_name in tqdm(enumerate(train_images), total=len(train_images)): 
        images = imread(train_axial_path_images + image_name)[:,:,:IMG_CHANNELS]
        images = resize(images, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        collective_images = np.maximum(collective_images, images)
    X_train[n] = collective_images  #Fill empty X_train with values from collective_images
    
    for k, mask_name in tqdm(enumerate(train_masks), total=len(train_masks)): 
        #we have change the image to gray scale and resize it ans adding new axis.
        mask = Image.open(train_axial_path_masks + mask_name)
        mask = mask.convert('L')
        mask = np.asarray(mask)
        mask = np.where((mask / 255) > 0.5, 1, 0)
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
        
        #mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        collective_mask = np.maximum(collective_mask, mask)
    Y_train[n] = collective_mask

    #checking the type the images, shape and the values of the array 
print(collective_images.shape)
print(collective_mask.shape)
print(X_train.shape)
print(Y_train.shape)
print(type(collective_images))
print(type(collective_mask))

#choose an random images and plot it 
image_x = random.randint(0, len(train_ids)-1)
imshow(X_train[image_x])
plt.show()

imshow(np.squeeze(Y_train[image_x]))
plt.show()

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

model = get_unet()

callbacks = EarlyStopping(patience=5, monitor='val_loss')
checkpointer = ModelCheckpoint('Model_Neu.h5', verbose=1, save_best_only=True)

print('*'*40)
print('Fitting Model')
print('*'*40)
history  = model.fit(X_train, Y_train, batch_size=4, epochs=50, shuffle=True,validation_split=0.1,callbacks=[checkpointer])

pyplot.title('Loss / BinaryCrossentropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

print(preds_train.shape)
print(preds_val.shape)
print(preds_test.shape)
 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print(preds_train_t.shape)
print(preds_val_t.shape)
print(preds_test_t.shape)



# Perform a sanity check on some random training samples
ix_train = random.randint(0, len(preds_train_t)-1)
print(ix_train)
imshow(X_train[ix_train])
plt.show()
imshow(np.squeeze(Y_train[ix_train]))
plt.show()
imshow(np.squeeze(preds_train_t[ix_train]))
plt.show()

# Perform a sanity check on some random validation samples
ix_val = random.randint(0, len(preds_val_t)-1)
print(ix_val)
imshow(X_train[int(X_train.shape[0]*0.9):][ix_val])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix_val]))
plt.show()
imshow(np.squeeze(preds_val_t[ix_val]))
plt.show()