# -*- coding: utf-8 -*-
#%%
#Loading Libraries
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

from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input,UpSampling2D

trainDataSet = pd.read_csv('driver_imgs_list.csv')

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
        files = glob(os.path.join( 'DataSet', 'train', 'c' + str(classed), '*.jpg'))
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
    path = os.path.join( 'DataSet', 'test', '*.jpg') #/DataSet/test/içerisinde tüm jpgleri alır
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

#---------Train data--------------
x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_width, img_height, color_type)
print('Train shape:', x_train.shape)
print(x_train.shape[0], 'Train Data sample')

#---------Test data---------------
test_sample_count = 100 #okunacak test veri sayısı
test_files, test_targets = read_and_normalize_test_data(test_sample_count, img_width, img_height, color_type) #rows_cols resmin boyutunu gönderiyor. color_type ise rgb mi greyscalemi
print('Test shape:', test_files.shape)
print(test_files.shape[0], 'Test Data sample') 

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))
x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(64, 64, 1))
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

plt.figure(figsize = (12, 20))
image_count = 1
url = 'DataSet/train/'
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
#%%
#//Model Construction
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
#//At this point the representation is (7, 7, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history=autoencoder.fit(x_train_noisy, x_train,
epochs=1,
batch_size=128,
shuffle=True,
validation_data=(x_test_noisy, x_test),
)
#%%save model
autoencoder.save_weights("autoencoder_model.h5")
#%%evaulation
print(history.history.keys())
#plt.plot(history.history["accuracy"],label="Train accuracy")
#plt.plot(history.history["val_acc"],label="validation accuracy")
plt.plot(history.history["loss"],label="Train loss",color="red")
plt.plot(history.history["val_loss"],label="validation loss")
plt.legend()
plt.show()

with open("autoencoders_hist.json","w") as f:
    json.dump(history.history,f)
    
with codecs.open("autoencoders_hist.json","r",encoding="utf-8") as f:
    n=json.loads(f.read())

print(n.keys())  
#plt.plot(n["accuracy"],label="Train accuracy")
#plt.plot(n["val_acc"],label="validation accuracy")  
plt.plot(n["loss"],label="Train loss")
plt.plot(n["val_loss"],label="validation loss")
#%%
def plot_train_history(history):
#    plt.plot(history["accuracy"])
#    plt.plot(history['val_accuracy'])
#    plt.title('Model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()

    # Summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
plot_train_history(n)

#%%
def predictImage(model, test_files, image_number, color_type=1):
    img = test_files[image_number]
    img = cv2.resize(img,(img_width,img_height))
    plt.imshow(img, cmap='gray')

    reshapedImg = img.reshape(-1,img_width,img_height,color_type)

    y_prediction = model.predict(reshapedImg, batch_size=128, verbose=1)
    print('Predicted: {}'.format(classes.get('c{}'.format(np.argmax(y_prediction)))))
    
    plt.show()
    
predictImage(autoencoder, test_files, 38)



































































#%% model
#input_img=Input(shape=(64,64,1))
#encoded=Dense(32,activation="relu")(input_img)
#encoded=Dense(16,activation="relu")(encoded)
#
#decoded=Dense(32,activation="relu")(encoded)
#decoded=Dense(784,activation="sigmoid")(decoded)
#
#autoencoder=Model(input_img,decoded)
#autoencoder.compile(optimizer="rmsprop",loss="binary_crossentropy")
#
#
#hist=autoencoder.fit(x_train_noisy,x_train,
#                     epochs=1,
#                     batch_size=128,
#                     shuffle=True,
#                     validation_data=(x_train_noisy,x_test))
#%%
