# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
from glob import glob
import random
import time
import tensorflow
import datetime
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

from tqdm import tqdm #Progress bar oluşturmak için kullanılmış sadece görüntüsü hoş
import json, codecs
import numpy as np
import pandas as pd
from IPython.display import FileLink
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 
#%matplotlib inline
from IPython.display import display, Image
import matplotlib.image as mpimg
import cv2

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files       
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

from keras import regularizers
from keras.optimizers import SGD
#from modelClass import ModelClass

class Klasa:
    def readImage(self,path):
        print(path)
        if self.color_type == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.color_type == 3:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_width, self.img_height))
        return img

    def predictImage(self,image):
        cv2.imwrite('../../images/testImage.jpg', image)
        img_brute = cv2.imread('../../images/testImage.jpg',1)

        im = cv2.resize(cv2.cvtColor(img_brute, cv2.COLOR_BGR2RGB), (self.img_width,self.img_height)).astype(np.float32) / 255.0
        im = np.expand_dims(im, axis =0)

        img_display = cv2.resize(img_brute,(self.img_width,self.img_height))
        plt.imshow(img_display, cmap='gray')

        y_preds = self.model.predict(im, batch_size=self.batch_size, verbose=1)
        print(y_preds)
        y_prediction = np.argmax(y_preds)
        print('Y Prediction: {}'.format(y_prediction))
        print('Predicted as: {}'.format(self.classes.get('c{}'.format(y_prediction))))
        plt.show()
        return self.classes.get('c{}'.format(y_prediction))
    
    def analyze(self,path):
        image=self.readImage(path)
        prediction=self.predictImage(image)
        return prediction

    def loadModel(self):
    # create the base pre-trained model
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.img_width, self.img_height, self.color_type))
        for layer in enumerate(base_model.layers):
            layer[1].trainable = False
    
        #flatten the results from conv block
        x = Flatten()(base_model.output)
    
        #add another fully connected layers with batch norm and dropout
        x = Dense(4096, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
    
        #add another fully connected layers with batch norm and dropout
        x = Dense(4096, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
    
    
        #add logistic layer with all car classes
        predictions = Dense(len(self.classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

    # this is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.load_weights('../../HistoryAndWeightFiles/vgg19_model_weights_v2.h5')

    def loadWeights(self):
        print("Called")
        
    def __init__(self):
        self.classes  = {'c0': 'Safe driving',
                        'c1': 'Texting - right',
                        'c2': 'Talking on the phone - right',
                        'c3': 'Texting - left',
                        'c4': 'Talking on the phone - left',
                        'c5': 'Operating the radio',
                        'c6': 'Drinking',
                        'c7': 'Reaching behind',
                        'c8': 'Hair and makeup',
                        'c9': 'Talking to passenger'}
    
        self.NUMBER_CLASSES = 10
        self.batch_size = 40
        self.img_width = 224
        self.img_height = 224
        self.color_type = 3
        self.model = None
        
        
