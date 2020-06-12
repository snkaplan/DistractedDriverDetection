# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
from glob import glob
import random
import time
import tensorflow as tf
graph = tf.get_default_graph()
import datetime
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

from tqdm import tqdm 
import json, codecs
import numpy as np
import pandas as pd
from IPython.display import FileLink
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, Image
import matplotlib.image as mpimg
import cv2
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

from keras import regularizers
from keras.optimizers import SGD

from keras.layers import Input

from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
class Klasa:
    def readImage(self,path):
        # # VGG19
        # print(path)
        # if self.color_type == 1:
        #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # elif self.color_type == 3:
        #     img = cv2.imread(path, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (self.img_width, self.img_height))
        # return img

        # ResNet50
        val_image = []
        if self.color_type == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.color_type == 3:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img[50:,120:-50]
            img = cv2.resize(img, (self.img_width, self.img_height))
            val_image.append(img)
            X_test = []
            for features in val_image:
                X_test.append(features)
            X_test = np.array(X_test).reshape(-1,224,224,3)
        return X_test[0]

    def predictImage(self,image):
        # # VGG-19
        # cv2.imwrite('../../images/testImage.jpg', image)
        # img_brute = cv2.imread('../../images/testImage.jpg',1)

        # im = cv2.resize(cv2.cvtColor(img_brute, cv2.COLOR_BGR2RGB), (self.img_width,self.img_height)).astype(np.float32) / 255.0
        # im = np.expand_dims(im, axis =0)

        # img_display = cv2.resize(img_brute,(self.img_width,self.img_height))
        # plt.imshow(img_display, cmap='gray')
        # with graph.as_default():
        #     y_preds = self.model.predict(im, batch_size=self.batch_size, verbose=1)
        # print(y_preds)
        # y_prediction = np.argmax(y_preds)
        # print('Y Prediction: {}'.format(y_prediction))
        # print('Predicted as: {}'.format(self.classes.get('c{}'.format(y_prediction))))
        # return self.classes.get('c{}'.format(y_prediction))

        # ResNet-50
        cv2.imwrite('../../images/testImage.jpg', image)
        img_brute = cv2.imread('../../images/testImage.jpg',1)
        img_display = cv2.resize(img_brute,(224,224))
        image = image.reshape(-1,224,224,3)
        y_preds = self.model.predict(image)
        print(y_preds)
        for pred in y_preds:
            y_prediction = np.argmax(pred)
            # print('Y Prediction: {}'.format(y_prediction))
            # print('Predicted as: {}'.format(classes.get('c{}'.format(y_prediction))))
        return self.classes.get('c{}'.format(y_prediction))
    def analyze(self,path):
        print(path)
        image=self.readImage(path)
        prediction=self.predictImage(image)
        return prediction

    def loadModel(self):
        # base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.img_width, self.img_height, self.color_type))
        # for layer in enumerate(base_model.layers):
        #     layer[1].trainable = False

        # x = Flatten()(base_model.output)

        # x = Dense(4096, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.5)(x)

        # x = Dense(4096, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.5)(x)


        # predictions = Dense(len(self.classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

        # self.model = Model(inputs=base_model.input, outputs=predictions)
        # self.model.load_weights('../../../HistoryAndWeightFiles/vgg19_model_weights_v2.h5')


        #ResNet-50

        resnet50_input = Input(shape = (224, 224, 3), name = 'Image_input')
        model_resnet50_conv = ResNet50(weights= 'imagenet', include_top=False, input_shape= (224,224,3))
        model_resnet50_conv.summary()
        from keras.models import Model
        output_resnet50_conv = model_resnet50_conv(resnet50_input)
        x=GlobalAveragePooling2D()(output_resnet50_conv)
        x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x = Dropout(0.1)(x) # **reduce dropout 
        x=Dense(1024,activation='relu')(x) #dense layer 2
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x=Dense(512,activation='relu')(x) #dense layer 3
        x = Dense(10, activation='softmax', name='predictions')(x)
        self.model = Model(input = resnet50_input, output = x)
        self.model.load_weights('../../../HistoryAndWeightFiles/resnet50(2).hdf5')

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

