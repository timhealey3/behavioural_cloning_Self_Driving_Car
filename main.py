import numpy as np 
import matplotlib.pyplot as py
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import pandas as pd 
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

datadir = 'data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(('data/driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', None)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

data['center'] = data['center'].apply(path_leaf)
data['leaf'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

data.head()
num_bins = 25
samples_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bins)
# balance data
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)

data.drop(data.index[remove_list], inplace=True)
hist, _ = np.histogram(data['steering'], num_bins)


def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_path = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

print(f"Training samples {X_train}")

print("Hello, world!")
