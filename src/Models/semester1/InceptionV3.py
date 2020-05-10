# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#%%Check GPU

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#%%
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
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split 
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D,  GlobalAveragePooling2D,Flatten, Dense, Dropout,BatchNormalization,Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense,Input
from keras.optimizers import SGD
from keras.regularizers import l2,l1,l1_l2
import keras
from keras.applications.inception_v3 import InceptionV3

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


img_width = 100 #64x64
img_height = 100
color_type = 3 #grey scale

#---------Train data--------------
#x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_width, img_height, color_type)
#print('Train shape:', x_train.shape)
#print(x_train.shape[0], 'Train Data sample')

#---------Test data---------------
test_sample_count = 100 #okunacak test veri sayısı
test_files, test_targets = read_and_normalize_test_data(test_sample_count, img_width, img_height, color_type) #rows_cols resmin boyutunu gönderiyor. color_type ise rgb mi greyscalemi
#print('Test shape:', test_files.shape)
#print(test_files.shape[0], 'Test Data sample') 
#%%
train_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.15, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale=1.0/ 255, validation_split = 0.2)

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
epoch = 100
input_image=Input(shape=(img_width,img_height,color_type))

def createModel():
    base_model=InceptionV3(weights='imagenet',include_top=False)

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x)
    x=Dense(256,activation='relu')(x)
    predictions=Dense(NUMBER_CLASSES,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=predictions)


    for layer in model.layers[249:]:
        layer.trainable=True
    
    
    return model
model = createModel()


model.summary()
#
##modeli hazırlar
#sgd = SGD(lr=0.001, momentum=0.9, decay=0.01, nesterov=True)
opt=keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
checkpoint = ModelCheckpoint('../HistoryAndWeightFiles/InceptionV3_model_weights_v6.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
#
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
##model bir değişkene fit edildi bu değişkenin değerleri kaydedilebilir
history = model.fit_generator(training_generator,
                         steps_per_epoch = nb_train_samples // batch_size,
                         epochs = epoch, 
                         callbacks=[es, checkpoint],
                         verbose = 1,
                         class_weight='balanced',
                         validation_data = validation_generator,
                         validation_steps = nb_validation_samples // batch_size)
  
#%% Model save 
#model.save_weights("CNN_Model_Weights.h5") ##modelin weights değişkenlerini kaydeder
#with open("./HistoryAndWeightFiles/CNN_Model_History.json","w") as f:  ##modelin accuracy değerlerini jsona yazar
#    json.dump(history.history,f) 

histt=pd.Series(history.history).to_json()
with open("../HistoryAndWeightFiles/InceptionV3_Model_History_v6.json","w") as f:  ##modelin accuracy değerlerini jsona yazar
    json.dump(histt,f) 
#%%Model Load
#model.load_weights('../HistoryAndWeightFiles/Indception_Model_Weights_v2.h5') #kayıtlı değişkenleri modele yükler

with codecs.open("../../HistoryAndWeightFiles/vgg16_model_history.json","r",encoding = "utf-8") as f: #accuracy ve lost değişkenlerini tekrar yükler
    oldHistory = json.loads(f.read())
print(oldHistory)
def plot_train_history(history):
    plt.plot(history["acc"])
    plt.plot(history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    # Summarize history for loss
#    plt.plot(history['loss'])
#    plt.plot(history['val_loss'])
#    plt.title('Model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
plot_train_history(oldHistory)
#%% Prediction Field
def predictImage(model, test_files, image_number, color_type=3):
    img = test_files[image_number]
    img = cv2.resize(img,(img_width,img_height))
    plt.imshow(img, cmap='gray')

    reshapedImg = img.reshape(-1,img_width,img_height,color_type)

    y_prediction = model.predict(reshapedImg, batch_size=batch_size, verbose=1)
    print('Predicted: {}'.format(classes.get('c{}'.format(np.argmax(y_prediction)))))
    
    plt.show()
    
predictImage(model, test_files,1)
