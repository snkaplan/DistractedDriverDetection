# -*- coding: utf-8 -*-
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
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

from keras import regularizers
from keras.optimizers import SGD
#%% LOAD IMAGES

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

NUMBER_CLASSES = 10 # toplam 10 tane kesin sınıfımız var
batch_size = 40
epoch = 100
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
    path = os.path.join( '../'+'DataSet', 'videoPhotos', '*.jpg') #/DataSet/test/içerisinde tüm jpgleri alır
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
        X_test_id.append(file_base)
        total += 1
    return X_test, X_test_id

#Test data yüklemesi
def read_and_normalize_test_data(size, img_width, img_height, color_type=3):
    test_data, test_ids = load_test(size, img_width, img_height, color_type)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(-1,img_width,img_height,color_type)

    return test_data, test_ids


img_width = 224
img_height = 224
color_type = 3


#---------Test data---------------
test_sample_count = 300
test_files, test_targets = read_and_normalize_test_data(test_sample_count, img_width, img_height, color_type)
print('Test shape:', test_files.shape)
print(test_files.shape[0], 'Test Data sample')

#%%Prediction and Accuracy Results
def plot_train_history(history):
    plt.plot(history["accuracy"])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predictImage(modelName,model, test_files, image_number):
    if(modelName=="CNN"):
        img = test_files[image_number]
        img = cv2.resize(img,(64,64))
        plt.imshow(img, cmap='gray')

        reshapedImg = img.reshape(-1,64,64,1)

        y_prediction = model.predict(reshapedImg, batch_size=batch_size, verbose=1)
        print('Predicted: {}'.format(classes.get('c{}'.format(np.argmax(y_prediction)))))

        plt.show()
        return y_prediction

    else:
        cv2.imwrite('./images/testImage.jpg', test_files[image_number])
        img_brute = cv2.imread('./images/testImage.jpg',1)

        im = cv2.resize(cv2.cvtColor(img_brute, cv2.COLOR_BGR2RGB), (img_width,img_height)).astype(np.float32) / 255.0
        im = np.expand_dims(im, axis =0)

        img_display = cv2.resize(img_brute,(img_width,img_height))
        plt.imshow(img_display, cmap='gray')

        y_preds = model.predict(im, batch_size=batch_size, verbose=1)
        print(y_preds)
        y_prediction = np.argmax(y_preds)
        print('Y Prediction: {}'.format(y_prediction))
        print(modelName+'Predicted as: {}'.format(classes.get('c{}'.format(y_prediction))))

        plt.show()
        return y_preds
#%%LOAD CNN MODEL
def createModel():
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = 3, padding='same', activation = 'relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters = 128, padding='same', kernel_size = 3, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters = 256, padding='same', kernel_size = 3, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters = 512, padding='same', kernel_size = 3, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUMBER_CLASSES, activation = 'softmax'))

    return model
modelCNN = createModel()

modelCNN.load_weights('./HistoryAndWeightFiles/CNN_Model_Weights.h5') #kayıtlı değişkenleri modele yükler
#
#with codecs.open("./HistoryAndWeightFiles/CNN_Model_History.json","r",encoding = "utf-8") as f: #accuracy ve lost değişkenlerini tekrar yükler
#    oldHistory = json.loads(f.read())


#plot_train_history(oldHistory)
#%%Load VGG16 Model
#vgg16 v2 2x(4096) lık dense layera sahip. O esnadaki en iyi ağırlıklara sahip.
def vgg_16_model(img_width, img_height, color_type=3):
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, color_type))
    for layer in enumerate(base_model.layers):
        layer[1].trainable = False

    x = Flatten()(base_model.output)

    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)




    predictions = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model

# Load the VGG16 network
print("Loading Model...")
model_vgg16 = vgg_16_model(img_width, img_height)


model_vgg16.load_weights('./HistoryAndWeightFiles/vgg16_model_weights.h5')
#
#with codecs.open("./HistoryAndWeightFiles/vgg16_model_history.json","r",encoding = "utf-8") as f:
#    oldHistory = json.loads(f.read())
#plot_train_history(oldHistory)
#%% VGG19 Model
#vgg19 v2 2x(4096) lık dense layera sahip. O esnadaki en iyi ağırlıklara sahip. vgg19v3 2x(4096) dense layer ve o esnadaki son weights
def vgg_19_model(img_width, img_height, color_type=3):
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, color_type))
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
    predictions = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

# Load the VGG16 network
print("Loading Model...")
model_vgg19 = vgg_19_model(img_width, img_height)


model_vgg19.load_weights('./HistoryAndWeightFiles/vgg19_model_weights_v2.h5')
#
#with codecs.open("./HistoryAndWeightFiles/vgg19_model_history.json","r",encoding = "utf-8") as f:
#    history = json.loads(f.read())

#%%
#predictImage("CNN ", modelCNN, test_files, 15)
#predictImage("VGG16 ",model_vgg16, test_files, 10)
prediction=predictImage("VGG19 ",model_vgg19, test_files,5)
#for i in range(88):
#    predictImage("VGG19 ",model_vgg19, test_files,i)
#    time.sleep(1)
