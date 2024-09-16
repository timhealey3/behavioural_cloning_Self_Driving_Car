import numpy as np 
import matplotlib.pyplot as py
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import pandas as pd 
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

datadir = 'Data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
os.chdir('..')
data = pd.read_csv(('Data/driving_log.csv'), names=columns)
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
    return image_path, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)
print(image_paths)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

def image_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

image = image_paths[100]
original_image = mpimg.imread(image)
original_image.show()
preprocessed_image = image_preprocess(original_image)

# pre process training && validation data 
X_train = np.array(list(map(image_preprocess, X_train)))
X_valid = np.array(list(map(image_preprocess, X_valid)))

# returns model object
def nvidia_model():
    model = Sequential()
    # pass normalized data to cnn 
    model.add(Convolution2D(24, 5, 5, stride=(2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    # use dropout layers to prevent overfitting
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100), activation='elu')
    model.add(Dropout(0.5))
    model.add(Dense(50), activation='elu')
    model.add(Dropout(0.5))
    model.add(Dense(10), activation='elu')
    model.add(Dropout(0.5))
    # single ouput node
    # output predicted steering angle 
    model.add(Dense(1))
    # compile model
    optimizer = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = nvidia_model()
print(model.summary())

# fit model
history = model.fix(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), batch_size=100, verbos=1, shuffle=1)

#model.save('model.h5')

print("Hello, world!")
