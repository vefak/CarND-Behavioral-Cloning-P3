from helpers import download, uncompress_features_labels
from helpers import  flipping, sharpen_img, augment_brightness_camera_images, trans_image

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D
from keras.models import Model
from skimage import io, color, exposure, filters, img_as_ubyte

import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
from urllib.request import urlretrieve
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from zipfile import ZipFile
import csv
import cv2

# Data provided by udacity
URL  = 'https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip'

path_to_download    = '/home/workspace/data.zip'
path_to_uncompress  = '/home/workspace/data/'
csvfile_path        = '/home/workspace/data/data/driving_log.csv'
image_path          = '/home/workspace/data/data/IMG/'

download(URL,path_to_download) 
uncompress_features_labels(path_to_download,path_to_uncompress)
 
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for idx,batch_sample in batch_samples.iterrows():
                for i in range(0,3):
                    if i==0:
                        corr = 0
                    elif i==1:
                        corr = 0.2
                    else:
                        corr = -0.2
                    #center image
                    center_img = image_path+batch_sample[i].split('/')[-1]      
                    center_image = mpimg.imread(center_img)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle+corr)
                    
                    #brightness
                    bright_image = augment_brightness_camera_images(center_image)
                    
                    images.append(bright_image)
                    angles.append(center_angle+corr)
                    
                    #translation 
                    #output = [image_tr,steer_ang,tr_x]
                    output = trans_image(center_image,float(batch_sample[3]),40)
                    images.append(output[0])
                    angles.append(output[1]+corr)
                    
                    #Flip
                    if(i==0):
                        flipped_img, flipped_ang = flipping(center_image, center_angle)
                        images.append(flipped_img)
                        angles.append(flipped_ang+corr)
                       
                    #Sharp
                    sharp_img = sharpen_img(center_image)
                    images.append(sharp_img)
                    angles.append(center_angle+corr)
          
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



drive_log = pd.read_csv(csvfile_path)
drive_log = drive_log[drive_log['steering'] != 0].append(drive_log[drive_log['steering'] == 0].sample(frac=0.5))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(drive_log, test_size=0.20)
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

# Parameters
batch_size = 128
epochs= 20

train_generator= generator(train_samples, batch_size=batch_size)
validation_generator= generator(validation_samples, batch_size=batch_size)

#Model. NVIDIA "End to End Learning for Self-Driving Cars" paper

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))           

#layer 1- Convolution, filters- 24, filter size= 5x5, stride= 2x2
model.add(Conv2D(24, (5, 5), subsample= (2,2), activation = 'elu'))

#layer 2- Convolution, filters- 36, filter size= 5x5, stride= 2x2
model.add(Conv2D(36, (5, 5), subsample= (2,2), activation = 'elu'))

#layer 3- Convolution, filters- 48, filter size= 5x5, stride= 2x2
model.add(Conv2D(48, (5, 5), subsample= (2,2), activation = 'elu'))

#layer 4- Convolution, filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3, 3), activation = 'elu'))

#layer 5- Convolution, filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3, 3), activation = 'elu'))

#Flatten
model.add(Flatten())

#fully connected layers
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
#model.add(Dropout(0.25))
model.add(Dense(10,activation='elu'))
model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification

#Compiling  Model
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
model.summary()

#Train model
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=int(np.ceil(len(train_samples)/batch_size)), 
            validation_data=validation_generator, 
            validation_steps=int(np.ceil(len(validation_samples)/batch_size)), 
            epochs=epochs, verbose=1)

model.save('model.h5')

