# -*- coding: utf-8 -*-

#### Bu hali 62. epochdan sonra düşüşe geçti 62 de 0.972 64'te 0.955 oldu 65'te 0.9653 devamında düşüşe devam etti en son 0.93 lerde kalacak
import os
from glob import glob
import time
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

from tqdm import tqdm #Progress bar oluşturmak için kullanılmış sadece görüntüsü var. İstersen kullanılan yerlerden kaldırırsın
import json, codecs
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.image as mpimg
import cv2

from sklearn.model_selection import train_test_split 
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

trainDataSet = pd.read_csv('../csvFiles/driver_imgs_list.csv')

#plt.figure(figsize = (10,10))
#sns.countplot(x = 'classname', data = trainDataSet)
#plt.ylabel('Count')
#plt.title('Categories')
#plt.show()



#%%
NUMBER_CLASSES = 10 # toplam 10 tane kesin sınıfımız var

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
        files = glob(os.path.join( '../','../','DataSet', 'train', 'c' + str(classed), '*.jpg'))
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
    path = os.path.join( '../', '../','DataSet', 'test', '*.jpg') #/DataSet/test/içerisinde tüm jpgleri alır
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



img_width = 64 #64x64
img_height = 64
color_type = 1 #grey scale
train_images,train_labels=load_train(img_width,img_height,color_type)
train_images = np.array(train_images, dtype=np.uint8).reshape(-1,img_width,img_height,color_type)
train_labels = np_utils.to_categorical(train_labels, 10)
#---------Train data--------------
x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_width, img_height, color_type)
print('Train shape:', x_train.shape)
print(x_train.shape[0], 'Train Data sample')

#---------Test data---------------
test_sample_count = 100 #okunacak test veri sayısı
test_files, test_targets = read_and_normalize_test_data(test_sample_count, img_width, img_height, color_type) #rows_cols resmin boyutunu gönderiyor. color_type ise rgb mi greyscalemi
print('Test shape:', test_files.shape)
print(test_files.shape[0], 'Test Data sample') 

#%% Her Gruptan birer örnek çizdirir

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
#%% MODEL Creation
batch_size = 40
epoch = 1          
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
cvscores = []
for train, test in kfold.split(train_images, train_labels):
    def createModel():
        model = Sequential() #sıralı model
    
        model.add(Conv2D(filters = 64, kernel_size = 3, padding='same', activation = 'relu', input_shape=(img_width, img_height, color_type)))
        model.add(MaxPooling2D())#default pool_size=2 gelir
    
        model.add(Conv2D(filters = 128, padding='same', kernel_size = 3, activation = 'relu'))
        model.add(MaxPooling2D())
    
        model.add(Conv2D(filters = 256, padding='same', kernel_size = 3, activation = 'relu'))
        model.add(MaxPooling2D()) 
    
        model.add(Conv2D(filters = 512, padding='same', kernel_size = 3, activation = 'relu'))
        model.add(MaxPooling2D())
    
        model.add(Dropout(0.5)) # her seferinde yarı yarı çıkaracak
    
        model.add(Flatten())
    
        model.add(Dense(500, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUMBER_CLASSES, activation = 'softmax'))
        
        return model
    model = createModel()
    
    #layerları gösterir
    #model.summary()
    #
    ##modeli hazırlar
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    ##model bir değişkene fit edildi bu değişkenin değerleri kaydedilebilir
    history = model.fit(train_images, train_labels, 
                        epochs=epoch, validation_data = (train_images, train_labels),
                        batch_size=batch_size, verbose=1)
#    scores = model.evaluate(train_images[test], train_labels[test], verbose=0)
#    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#    cvscores.append(scores[1] * 100)
#print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#%% Model save 
#model.save_weights("CNN_Model_Weights.h5") ##modelin weights değişkenlerini kaydeder
#with open("./HistoryAndWeightFiles/CNN_Model_History.json","w") as f:  ##modelin accuracy değerlerini jsona yazar
#    json.dump(history.history,f) 
#histt=pd.Series(history.history).to_json()
#with open("./HistoryAndWeightFiles/CNN_Model_History3.json","w") as f:  ##modelin accuracy değerlerini jsona yazar
#    json.dump(histt,f) 
#%%Model Load
model.load_weights('../HistoryAndWeightFiles/CNN_Model_Weights.h5') #kayıtlı değişkenleri modele yükler
#
with codecs.open("../HistoryAndWeightFiles/CNN_Model_History.json","r",encoding = "utf-8") as f: #accuracy ve lost değişkenlerini tekrar yükler
    oldHistory = json.loads(f.read())

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
plot_train_history(oldHistory)
#%% Prediction Field
def predictImage(model, test_files, image_number, color_type=1):
    img = test_files[image_number]
    img = cv2.resize(img,(img_width,img_height))
    plt.imshow(img, cmap='gray')

    reshapedImg = img.reshape(-1,img_width,img_height,color_type)

    y_prediction = model.predict(reshapedImg, batch_size=batch_size, verbose=1)
    print('Predicted: {}'.format(classes.get('c{}'.format(np.argmax(y_prediction)))))
    
    plt.show()
    
predictImage(model, test_files, 33)