
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


def download(url, path):
    if not os.path.isfile(path):
        urlretrieve(url,path)
        print("File downloaded")
    else:
        print("file already downloaded")

def uncompress_features_labels(dir,path):
    if(os.path.exists(path)):
        print('Data already extracted')
    else:
        with ZipFile(dir) as zipf:
            zipf.extractall(path)
            print("Data extracted.")

def flipping(img, angle):
    flipped = cv2.flip(img, 1)
    flip_ang = angle * -1
    return flipped, flip_ang

def sharpen_img(img):

    gb  = cv2.GaussianBlur(img, (7,7), 15.0)
    shp = cv2.addWeighted(img, 2, gb, -1, 0)
    return shp.reshape(160,320,3)

       
def augment_brightness_camera_images(img):
    image = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def trans_image(image,steer,trans_range):
    # Translation
    rows,cols,channels = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image_tr,steer_ang,tr_x