import os
from glob import glob
import time
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

from tqdm import tqdm 
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






class Model:
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
    self.model = self.loadModel()
    self.model = self.loadWeights()
    
    def analyze(self,path):
        image=readImage(path)
        prediction=predictImage(image)
        return prediction
        

    def readImage(self,path):
        if self.color_type == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.color_type == 3:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_width, self.img_height)) 
        return img
    
    def predictImage(self,image):
        cv2.imwrite('./images/testImage.jpg', image)
        img_brute = cv2.imread('./images/testImage.jpg',1)
    
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
    
    def loadModel(self):
        vgg16_model = VGG16(weights="imagenet", include_top=False,  input_shape=(self.img_width, self.img_height, self.color_type))
        
        vgg_layer_list=vgg16_model.layers
        
        model=Sequential()
        
        for layer in vgg_layer_list:
            model.add(layer)
            
        for layer in model.layers:
            layer.trainable=False
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUMBER_CLASSES,activation="softmax"))
        
        return model
    
    def loadWeights(self):
        self.model.load_weights('../HistoryAndWeightFiles/vgg16_model_weights.h5')