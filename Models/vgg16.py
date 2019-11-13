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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16

trainDataSet = pd.read_csv('../csvFiles/driver_imgs_list.csv')

plt.figure(figsize = (10,10))
sns.countplot(x = 'classname', data = trainDataSet)
plt.ylabel('Count')
plt.title('Categories')
plt.show()



#%%
NUMBER_CLASSES = 10 # toplam 10 tane kesin sınıfımız var

def get_cv2_image(path, img_width, img_height, color_type=3):
    if color_type == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (img_width, img_height)) # resmi model için 64x64 boyutuna indirgiyoruz
    return img

## Train klasöründeki tüm resimleri okuyor
#def load_train(img_width, img_height, color_type=3):
#    start_time = time.time()
#    train_images = [] 
#    train_labels = []
#    # Loop over the training folder 
#    for classed in tqdm(range(NUMBER_CLASSES)):
#        print('Loading directory c{}'.format(classed))
#        files = glob(os.path.join( '../','../','DataSet', 'train', 'c' + str(classed), '*.jpg'))
#        for file in files:
#            img = get_cv2_image(file, img_width, img_height, color_type)
#            train_images.append(img)
#            train_labels.append(classed)
#    print("Data Loaded in {} second".format(time.time() - start_time))
#    return train_images, train_labels 
#
##Train data yüklemesi yapıyor
#def read_and_normalize_train_data(img_width, img_height, color_type):
#    train_images, train_labels = load_train(img_width, img_height, color_type)
#    y = np_utils.to_categorical(train_labels, 10)
#    x_train, x_test, y_train, y_test = train_test_split(train_images, y, test_size=0.2, random_state=42)
#    
#    x_train = np.array(x_train, dtype=np.uint8).reshape(-1,img_width,img_height,color_type)
#    x_test = np.array(x_test, dtype=np.uint8).reshape(-1,img_width,img_height,color_type)
#    
#    return x_train, x_test, y_train, y_test

# Test klasöründeki tüm resimleri okuyacak
def load_test(size=200000, img_width=64, img_height=64, color_type=3):
    path = os.path.join( '../','../','DataSet', 'test', '*.jpg')
    files = sorted(glob(path))
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


img_width = 224 #64x64
img_height = 224
color_type = 3 #rgb scale

#---------Train data--------------
#x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_width, img_height, color_type)
#print('Train shape:', x_train.shape)
#print(x_train.shape[0], 'Train Data sample')
#
##---------Test data---------------
test_sample_count = 100 #okunacak test veri sayısı
test_files, test_targets = read_and_normalize_test_data(test_sample_count, img_width, img_height, color_type) #rows_cols resmin boyutunu gönderiyor. color_type ise rgb mi greyscalemi
print('Test shape:', test_files.shape)
print(test_files.shape[0], 'Test Data sample') 

#%%

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

batch_size = 40
epoch = 8                


plt.figure(figsize = (12, 20))
image_count = 1
url = '../../DataSet/train/'
for directory in os.listdir(url): #url içindeki tüm klasörlerde gezecek
    if directory[0] != '.':
        for i, file in enumerate(os.listdir(url + directory)):
            if i == 1:
                break
            else:
                fig = plt.subplot(5, 2, image_count)
                image_count += 1
                images = mpimg.imread(url + directory + '/' + file) #mpimg matplotlib içinde image göstermeye yarayan kütüphane
                plt.imshow(images) #images değişkenini çizdi
                plt.title(classes[directory]) #başlığına classes içinden kendi grubunu koydu

#%% VGG16----
#%%Data Generator-------------
# Prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.15, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale=1.0/ 255, validation_split = 0.2)
#%% Model
def vgg_16_model(img_width, img_height, color_type=3):
    NumberOfClass = 10
    vgg16_model = VGG16(weights="imagenet", include_top=False,  input_shape=(img_width, img_height, color_type))
    
    vgg_layer_list=vgg16_model.layers
    
    model=Sequential()
    
    for layer in vgg_layer_list:
        model.add(layer)
        
    for layer in model.layers:
        layer.trainable=False
    #fully connected layer
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(NumberOfClass,activation="softmax"))
    
    return model

# Load the VGG16 network
print("Loading Model...")
model_vgg16 = vgg_16_model(img_width, img_height)

model_vgg16.summary()

model_vgg16.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])

training_generator = train_datagen.flow_from_directory('../../DataSet/train', 
                                                 target_size = (img_width, img_height), 
                                                 batch_size = batch_size,
                                                 shuffle=True,
                                                 class_mode='categorical', subset="training")

validation_generator = test_datagen.flow_from_directory('../../DataSet/train', 
                                                   target_size = (img_width, img_height), 
                                                   batch_size = batch_size,
                                                   shuffle=False,
                                                   class_mode='categorical', subset="validation")
nb_train_samples = 17943
nb_validation_samples = 4481
#
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
checkpoint = ModelCheckpoint('../HistoryAndWeightFiles/vgg16_model_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model_vgg16.fit_generator(training_generator,
                         steps_per_epoch = nb_train_samples // batch_size,
                         epochs = epoch, 
                         callbacks=[es, checkpoint],
                         verbose = 1,
                         class_weight='balanced',
                         validation_data = validation_generator,
                         validation_steps = nb_validation_samples // batch_size)

#%% Model save      
histt=pd.Series(history.history).to_json()
with open("../HistoryAndWeightFiles/vgg16_model_history.json","w") as f:  ##modelin accuracy değerlerini jsona yazar
    json.dump(histt,f) 


#%% Load history and wights
#model_vgg16.load_weights('../HistoryAndWeightFiles/vgg16_model_weights.h5')
#
with codecs.open("../HistoryAndWeightFiles/vgg16_model_history.json","r",encoding = "utf-8") as f:
    oldHistory = json.loads(f.read())
def plot_train_history(history):
    plt.plot(history['accuracy'])
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
plot_train_history(oldHistory)

#%% prediction
def plot_vgg16_test_class(model, test_files, image_number):
    cv2.imwrite('../images/testImageVGG16.jpg', test_files[image_number])
    img_brute = cv2.imread('../images/testImageVGG16.jpg',1)

    im = cv2.resize(cv2.cvtColor(img_brute, cv2.COLOR_BGR2RGB), (img_width,img_height)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis =0)

    img_display = cv2.resize(img_brute,(img_width,img_height))
    plt.imshow(img_display, cmap='gray')

    y_preds = model.predict(im, batch_size=batch_size, verbose=1)
    print(y_preds)
    y_prediction = np.argmax(y_preds)
    print('Y Prediction: {}'.format(y_prediction))
    print('Predicted as: {}'.format(classes.get('c{}'.format(y_prediction))))
    
    plt.show()
    
plot_vgg16_test_class(model_vgg16, test_files,8) # Texting left 80 66 10 8


#score = model_vgg16.evaluate_generator(validation_generator, nb_validation_samples // batch_size, verbose = 1)
#print("Test Score:", score[0])
#print("Test Accuracy:", score[1])
