import os
import numpy as np
import keras
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from PIL import Image
from functools import partial
import sys
from keras import optimizers

def get_LeNet5_net(labels_num):
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(5, 5), filters=32, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
    model.add(BatchNormalization()) # BN is good 
    model.add(Conv2D(kernel_size=(5, 5), filters=64,  activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
    model.add(BatchNormalization()) # BN is useful
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(labels_num, activation='softmax'))
    return model


