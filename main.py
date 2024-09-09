import numpy as np 
import matplotlib.pyplot as py
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import pandas as pd 
import random

datadir = 'data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
pd.read_csv(('data/driving_log.csv'), names=columns)

print("Hello, world!")
