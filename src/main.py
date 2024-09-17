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
from imgaug import augmenters as iaa 

# returns numpy.ndarray of all image paths
# and numpy.ndarray of all steering angles
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

# data augmentation image generator
# returns numpy.ndarray of image
def zoom(img):
    # zoom in up to 30% 
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(img)
    return image

# returns numpy.ndarray of image
def pan(img):
    # pan up to 10% on both x and y axis
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(img)
    return image

# returns numpy.ndarray of image
def img_random_brightness(img):
    # change brightness of pixels
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(img)
    return image

# returns numpy.ndarray of image
# and numpy.float64 of steering_angle
def flipping(img, steering_angle):
    # flip some images and steering angle
    image = cv2.flip(img, 1)
    steering_angle = -steering_angle
    return image, steering_angle

# returns numpy.ndarray of image
# and numpy.float64 of steering_angle
def random_augment(img, steering_angle):
    image = mpimg.imread(img)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flipping(image, steering_angle)
    return image, steering_angle

# returns numpy.ndarray of preprocessed image
def image_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def batch_generator(image_paths, steering_angle, batch_size, isTrainingInd):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            if isTrainingInd:
                img, steering = random_augment(image_paths[random_index], steering_angle[random_index])
            else:
                img = mpimg.imread(image_paths[random_index])
                steering = steering_angle[random_index]
            img = image_preprocess(img)
            batch_img.append(img)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))

# Neural Network based on the nvidia model for self driving cars
# returns model object
def nvidia_model():
    model = Sequential()
    # pass normalized data to cnn 
    model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    # single ouput node
    # output predicted steering angle 
    model.add(Dense(1))
    # compile model
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

if __name__ == "__main__":
    # loading data
    datadir = 'track'
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    os.chdir('..')
    data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
    pd.set_option('display.max_colwidth', None)
    data.head()
    
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail

    data['center'] = data['center'].apply(path_leaf)
    data['leaf'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)

    data.head()
    num_bins = 25
    samples_per_bin = 400
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
    
    image_paths, steerings = load_img_steering(datadir + '/IMG', data)
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

    # call image generator
    X_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
    X_train_gen, y_train_gen = next(batch_generator(X_valid, y_valid, 1, 0))

    model = nvidia_model()
    print(model.summary())

    # fit model
    history = model.fit(
        batch_generator(X_train, y_train, 100, 1), 
        steps_per_epoch=300, 
        epochs=10, 
        validation_data=batch_generator(X_valid, y_valid, 100, 0),
        validation_steps=200, 
        verbose=1, 
        shuffle=1
    )

    model.save('src/model.keras')
