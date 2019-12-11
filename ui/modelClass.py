import os
from glob import glob
import time
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

import json, codecs
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16


class ModelClass(object):
  def __init__(self):
    self.NUMBER_CLASSES = 10


    def loadModel(self):
        model = VGG16(weights="imagenet", include_top=False,  input_shape=(self.img_width, self.img_height, self.color_type))

        vgg_layer_list=model.layers

        model=Sequential()

        for layer in vgg_layer_list:
            model.add(layer)

        for layer in self.model.layers:
            layer.trainable=False
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUMBER_CLASSES,activation="softmax"))
        model.load_weights('../HistoryAndWeightFiles/vgg16_model_weights.h5')
        return model
        
        
