import os.path
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from natsort import natsorted, ns
import glob

seed = 42
np.random.seed = seed
np.random.seed = seed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 64
IMG_CHANNELS = 3

TRAIN_PATH = "B:/Masterarbeit/New life/Dataset/IVDM_Update/train/"
TEST_PATH = "B:/Masterarbeit/New life/Dataset/IVDM_Update/test/"
mask_Path = "B:/Masterarbeit/New life/Dataset/IVDM_Update/train/"

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
mask_ids = next(os.walk(mask_Path))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
#type of boolen if True or False
Y_train = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Resizing test images') 

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    test_path = TEST_PATH + id_ + '/Fat/' +'Axial/'
    test_images = next(os.walk(test_path))[2]
    test_images = natsorted(test_images, alg=ns.IGNORECASE)
    collective_test_images = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    #print(test_images)
    for m, test_image_name in tqdm(enumerate(test_images), total=len(test_images)):
        test_img = imread(test_path + test_image_name)[:,:,:IMG_CHANNELS]
        test_img = resize(test_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        collective_test_images = np.maximum(collective_test_images, test_img)
    X_test[n] = collective_test_images

print(type(collective_test_images))

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

        mask = Image.open(train_axial_path_masks + mask_name)
        mask = mask.convert('L')
        mask = np.asarray(mask)
        mask = np.where((mask / 255) > 0.5, 1, 0)
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
        
        #mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        collective_mask = np.maximum(collective_mask, mask)
    Y_train[n] = collective_mask
#to run faster without plotting the images and it's mask 

print(collective_images.shape)
print(collective_mask.shape)
print(X_train.shape)
print(Y_train.shape)
print(type(collective_images))
print(type(collective_mask))


image_x = random.randint(0, len(train_ids)-1)
imshow(X_train[image_x])
plt.show()

imshow(np.squeeze(Y_train[image_x]))
plt.show()


inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 



#### Dice Coff ####### 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss=dice_loss , metrics=['accuracy'])
model.summary()

##print(len(X_train))

################################
#Modelcheckpoint#
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

#callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
             #tf.keras.callbacks.TensorBoard(log_dir='logs')]
        
        
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=4, epochs=15, callbacks=[checkpointer])


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



