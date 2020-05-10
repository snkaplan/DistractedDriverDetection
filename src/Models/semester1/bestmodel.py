# utility fxn to plot model history and accuracy for each epoch
import numpy as np 
import pandas as pd 
import os
from keras import layers
from keras import models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from os import listdir, makedirs
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, VGG19, InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers, regularizers
from keras.optimizers import SGD
from glob import glob
import cv2
import warnings 
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from mpl_toolkits.axes_grid1 import ImageGrid

import tensorflow as tf
import time
import os
#from tqdm import tqdm # for progress indication

#print(os.listdir("../input"))
#data_dir = '../input/'
#data_dir1= '../input/state-farm-distracted-driver-detection/'

#def plot_model_history(model_history):
#    fig, axs = plt.subplots(1,2,figsize=(15,5))
#    # summarize history for accuracy
#    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
#    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
#    axs[0].set_title('Model Accuracy')
#    axs[0].set_ylabel('Accuracy')
#    axs[0].set_xlabel('Epoch')
#    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
#    axs[0].legend(['train', 'val'], loc='best')
#    # summarize history for loss
#    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
#    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
#    axs[1].set_title('Model Loss')
#    axs[1].set_ylabel('Loss')
#    axs[1].set_xlabel('Epoch')
#    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
#    axs[1].legend(['train', 'val'], loc='best')
#    plt.show()
#    
## utiliy fxn to get y_predict in 1D
## y_predict is array of 12 classes for each cases.. let form the new data which give label value in 1D.. 
## this is required for classification matrix.. cm expect 1D array
#def get1D_y_predict(y_pred):
#    result = []
#    for i in range(len(y_pred)):
#        result.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
#    return result    
#
#def plot_cnf_matrix(cnf_matrix, name):
#    fig, ax = plt.subplots(1, figsize=(12,5))
#    ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
#    ax.set_xticklabels(class_list)
#    ax.set_yticklabels(class_list)
#    plt.title('Confusion Matrix')
#    plt.ylabel('True class')
#    plt.xlabel('Predicted class')
#    fig.savefig('{}_cnf.png'.format(name), dpi=300)
#    plt.show();
    
# use tensorboard callback which will passed in model.fit function.
# utility fxn ffor Initializing Early stopping and Model chekpoint callbacks**
#def EarlyStopingModelCheckPoint():
#    #tensorboard = TensorBoard(log_dir=".logs/{}".format(time.time()))
#
#    #Adding Early stopping callback to the fit function is going to stop the training,
#    #if the val_loss is not going to change even '0.001' for more than 5 continous epochs
#
#    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)
#
#    #Adding Model Checkpoint callback to the fit function is going to save the weights whenever val_loss achieves 
#    # a new low value. Hence saving the best weights occurred during training
#
#    model_checkpoint =  ModelCheckpoint('bestmodel.h5',
#                                                               monitor='val_loss',
#                                                               verbose=1,
#                                                               save_best_only=True,
#                                                               save_weights_only=False,
#                                                               mode='auto',
#                                                               period=1)
#    return early_stopping, model_checkpoint
def create_model_resnet():
    resnet = ResNet50(include_top=False, input_shape=(224, 224, 3))
    
    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(Dense(10, activation='softmax'))   
    model.summary()
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    return resnet, model
def create_my_resnet(resnet):    
    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(Dense(10, activation='softmax'))   
    model.summary()
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    return  model

resnet = ResNet50(include_top=False, input_shape=(128, 128, 3))
resnet.summary()
model_resnet = create_my_resnet(resnet)


NUMBER_CLASSES = 10 # toplam 10 tane kesin sınıfımız var
batch_size = 64
classes      = {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}
def predictImage(modelName,model, test_files, image_number):
    img = test_files[image_number]
    img = cv2.resize(img,(img_width,img_height))
    plt.imshow(img, cmap='gray')

    reshapedImg = img.reshape(-1,img_width,img_height,color_type)

    y_prediction = model.predict(reshapedImg, batch_size=batch_size, verbose=1)
    print('Predicted: {}'.format(classes.get('c{}'.format(np.argmax(y_prediction)))))
    
    plt.show()
    
model_resnet.load_weights('./HistoryAndWeightFiles/bestmodel.h5') #kayıtlı değişkenleri modele yükler




from tqdm import tqdm #Progress bar oluşturmak için kullanılmış sadece görüntüsü hoş
import json, codecs
import numpy as np
import pandas as pd
from keras.utils import np_utils

def get_cv2_image(path, img_width, img_height, color_type=3):
    if color_type == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height)) # resmi cnn için 64x64 boyutuna indirgiyoruz
    return img

# Train klasöründeki tüm resimleri okuyor
def load_train(img_width, img_height, color_type=3):
    start_time = time.time()
    train_images = [] 
    train_labels = []
    # Loop over the training folder 
    for classed in tqdm(range(NUMBER_CLASSES)):
        print('Loading directory c{}'.format(classed))
        files = glob(os.path.join( '../'+'DataSet', 'train', 'c' + str(classed), '*.jpg'))
        for file in files:
            img = get_cv2_image(file, img_width, img_height, color_type)
            train_images.append(img)
            train_labels.append(classed)
    print("Data Loaded in {} second".format(time.time() - start_time))
    return train_images, train_labels 

#Train data yüklemesi yapıyor
def read_and_normalize_train_data(img_width, img_height, color_type):
    train_images, train_labels = load_train(img_width, img_height, color_type)
    y = np_utils.to_categorical(train_labels, 10)
    x_train, x_test, y_train, y_test = train_test_split(train_images, y, test_size=0.2, random_state=42)
    
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1,img_width,img_height,color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1,img_width,img_height,color_type)
    
    return x_train, x_test, y_train, y_test

# Test klasöründeki tüm resimleri okuyacak
def load_test(size=200000, img_width=64, img_height=64, color_type=3):
    path = os.path.join( '../'+'DataSet', 'test', '*.jpg') #/DataSet/test/içerisinde tüm jpgleri alır
    files = sorted(glob(path)) #topladığı dosyaları sıraladı 79000 fotoğraf
    X_test, X_test_id = [], [] 
    total = 0
    files_size = len(files)
    for file in tqdm(files):
        if total >= size or total >= files_size:
            break
        file_base = os.path.basename(file)
        img = get_cv2_image(file, img_width, img_height, color_type)
        X_test.append(img)
        X_test_id.append(file_base) #Bu iki satır dosyaları ve dosya adlarını kaydediyor bir arraye
        total += 1
    return X_test, X_test_id

#Test data yüklemesi
def read_and_normalize_test_data(size, img_width, img_height, color_type=3):
    test_data, test_ids = load_test(size, img_width, img_height, color_type)
    
    test_data = np.array(test_data, dtype=np.uint8) #image dosyaların array hali uint8 olarak kullanılmış çoğu kernelde
    test_data = test_data.reshape(-1,img_width,img_height,color_type)
    
    return test_data, test_ids


img_width = 128 #64x64
img_height = 128
color_type = 3 #grey scale

#---------Train data--------------
#x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_width, img_height, color_type)
#print('Train shape:', x_train.shape)
#print(x_train.shape[0], 'Train Data sample')

#---------Test data---------------
test_sample_count = 300 #okunacak test veri sayısı
test_files, test_targets = read_and_normalize_test_data(test_sample_count, img_width, img_height, color_type) #rows_cols resmin boyutunu gönderiyor. color_type ise rgb mi greyscalemi
#print('Test shape:', test_files.shape)
#print(test_files.shape[0], 'Test Data sample') 

predictImage("Res ", model_resnet, test_files, 299)
