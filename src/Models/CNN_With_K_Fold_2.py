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
from sklearn.model_selection import train_test_split ,StratifiedKFold,GroupKFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,  GlobalAveragePooling2D,Flatten, Dense, Dropout,BatchNormalization,Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import l2,l1,l1_l2

trainDataSet = pd.read_csv('../csvFiles/driver_imgs_list.csv')
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
#        print('Loading directory c{}'.format(classed))
        files = glob(os.path.join( '../','../','DataSet', 'train', 'c' + str(classed), '*.jpg'))
        for file in files:
            img = get_cv2_image(file, img_width, img_height, color_type)
            train_images.append(img)
            train_labels.append(classed)
#    print("Data Loaded in {} second".format(time.time() - start_time))
    return train_images, train_labels 

#Train data yüklemesi yapıyor
def read_and_normalize_train_data(img_width, img_height, color_type):
    train_images, train_labels = load_train(img_width, img_height, color_type)
    y = np_utils.to_categorical(train_labels, 10)
    x_train, x_test, y_train, y_test = train_test_split(train_images, y, test_size=0.2, random_state=42)
    
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1,img_width,img_height,color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1,img_width,img_height,color_type)
    
    return x_train, x_test, y_train, y_test

def load_data_kfold(k):
    x_train, x_test, y_train, y_test=read_and_normalize_train_data(img_width, img_height, color_type)
#    folds=[]
#    folds=list(StratifiedKFold(n_splits=k,shuffle=True,random_state=42).split(x_train,y_train,groups=10))
    folds= list(GroupKFold(n_splits=k).split(x_train,y_train,groups=np.array(classes) ))
    
    
   
    return folds,x_train,y_train
    

img_width = 100 #64x64
img_height = 100
color_type = 3 #grey scale
k=10
folds,x_train,y_train=load_data_kfold(k)

test_sample_count = 100 #okunacak test veri sayısı
test_files, test_targets = read_and_normalize_test_data(test_sample_count, img_width, img_height, color_type) #rows_cols resmin boyutunu gönderiyor. color_type ise rgb mi greyscalemi
#print('Test shape:', test_files.shape)
#print(test_files.shape[0], 'Test Data sample') 
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
#%%
batch_size = 40
epoch = 1


def createModel():
    model=Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),input_shape=(img_width, img_height, color_type))) #imageler 2D olduğu için Conv da 2D--- 32 tane filtre olacak aynı zamanda 32 feature map
                                #imagelerın boyutu 3,3 lük bir matris input_shape başta bulununan resimlerin shapei
    model.add(Activation("relu")) #relu aktivasyon fonksiyonu eklendi
    model.add(MaxPooling2D()) #defaultu 2,2 lik olduğu için bıraktık
    
    
    
    model.add(Conv2D(64,(3,3),input_shape=(img_width, img_height, color_type))) #imageler 2D olduğu için Conv da 2D--- 32 tane filtre olacak aynı zamanda 32 feature map
                                #imagelerın boyutu 3,3 lük bir matris input_shape başta bulununan resimlerin shapei
    model.add(Activation("relu")) #relu aktivasyon fonksiyonu eklendi
    model.add(MaxPooling2D()) #defaultu 2,2 lik olduğu için bıraktık
    
#    
    model.add(Conv2D(128,(3,3),input_shape=(img_width, img_height, color_type))) #imageler 2D olduğu için Conv da 2D--- 32 tane filtre olacak aynı zamanda 32 feature map
                                #imagelerın boyutu 3,3 lük bir matris input_shape başta bulununan resimlerin shapei
    model.add(Activation("relu")) #relu aktivasyon fonksiyonu eklendi
    model.add(MaxPooling2D()) #defaultu 2,2 lik olduğu için bıraktık
    
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),input_shape=(img_width, img_height, color_type))) #imageler 2D olduğu için Conv da 2D--- 32 tane filtre olacak aynı zamanda 32 feature map
                                #imagelerın boyutu 3,3 lük bir matris input_shape başta bulununan resimlerin shapei
    model.add(Activation("relu")) #relu aktivasyon fonksiyonu eklendi
    model.add(GlobalAveragePooling2D()) #defaultu 2,2 lik olduğu için bıraktık
    
    model.add(Flatten())
#    model.add(Dense(1024)) #1024 tane nörondan oluşacak
#    model.add(Activation("relu"))
    
    
    model.add(Dense(NUMBER_CLASSES))#output class class sayısı başta bulunan class sayısı olmalı
#    model.add(Dropout(0.3)) #yani her tekrarda nöronların yarısı kapanacak makinenin ezberlemesi önlenecek

    model.add(Activation("softmax")) #en son kullanılan softmax aktivasyonu eklendi
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
model = createModel()

#layerları gösterir
model.summary()

gen = ImageDataGenerator(horizontal_flip = True,
                         rescale = 1.0/255, 
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.15,
                         shear_range = 0.2, 
                         validation_split = 0.2,
                         rotation_range = 10)

test_datagen = ImageDataGenerator(rescale=1.0/ 255, validation_split = 0.2)

def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]

nb_train_samples = 17943
nb_validation_samples = 4481
for j, (train_idx, val_idx) in enumerate(folds):
    
    print('\nFold ',j)
    X_train_cv = x_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = x_train[val_idx]
    y_valid_cv= y_train[val_idx]
    
    
    callbacks = get_callbacks('../HistoryAndWeightFiles/CNN_KFold_model_weights_v1.h5', patience_lr=10)
    generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
    model = get_model()
    history=model.fit_generator(
                generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epoch,
                shuffle=True,
                verbose=1,
                validation_data = (X_valid_cv, y_valid_cv),
                callbacks = callbacks)
    
    print(model.evaluate(X_valid_cv, y_valid_cv))


histt=pd.Series(history.history).to_json()
with open("../HistoryAndWeightFiles/CNN_KFold_Model_History_v1.json","w") as f:  ##modelin accuracy değerlerini jsona yazar
    json.dump(histt,f) 
    

with codecs.open("../HistoryAndWeightFiles/CNN_KFold_Model_History_v1.json","r",encoding = "utf-8") as f: #accuracy ve lost değişkenlerini tekrar yükler
    oldHistory = json.loads(f.read())


#%% Prediction Field
def predictImage(model, test_files, image_number, color_type=3):
    img = test_files[image_number]
    img = cv2.resize(img,(img_width,img_height))
    plt.imshow(img, cmap='gray')

    reshapedImg = img.reshape(-1,img_width,img_height,color_type)

    y_prediction = model.predict(reshapedImg, batch_size=batch_size, verbose=1)
    print('Predicted: {}'.format(classes.get('c{}'.format(np.argmax(y_prediction)))))
    
    plt.show()
    
predictImage(model, test_files,2)













