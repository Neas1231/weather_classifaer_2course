#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import glob
import tensorflow  as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import metrics
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = 'C:\НЕ УДАЛЯТЬ НИКОГДА НИ ПРИ КАКИХ УСЛОВИЯХ !!!!\dataset'
path_imgs = list(glob.glob(path+'/**/*.jpg'))

get_ipython().run_line_magic('matplotlib', 'inline')
labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1], path_imgs))
file_path = pd.Series(path_imgs, name='File_Path').astype(str)
labels = pd.Series(labels, name='Labels')
data = pd.concat([file_path, labels], axis=1)
data = data.sample(frac=1).reset_index(drop=True)
data.head()


# In[2]:


dictuar = {'dew':0 ,'fogsmog':1 ,'frost':2 ,'glaze':3 ,'hail':4 ,'lightning':5 , 'rain':6, 'rainbow':7, 
           'rime':8, 'sandstorm':9, 'snow':10}


# In[3]:


train_df, labels_test = train_test_split(data["Labels"], test_size=0.33, random_state=0)
datagen= keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.33, #Split 75% for train and 25% for validation/test
    rescale=1./255 #Rescale the images
)
train_ds = datagen.flow_from_directory(
    'C:\НЕ УДАЛЯТЬ НИКОГДА НИ ПРИ КАКИХ УСЛОВИЯХ !!!!\dataset',
    target_size=(256, 256), #Target size
    batch_size=32,
    class_mode='categorical',
    subset='training') # set as training data


val_ds = datagen.flow_from_directory(
    'C:\НЕ УДАЛЯТЬ НИКОГДА НИ ПРИ КАКИХ УСЛОВИЯХ !!!!\dataset', # same directory as training data
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation') # set as validation data


# In[4]:


import random

fig, axes = plt.subplots(3,3, figsize=(20, 15))

indices_classes = {v: k for k, v in train_ds.class_indices.items()} #Словарь классов
images_classes = list(zip(train_ds.filepaths, [indices_classes[k] for k in train_ds.classes])) #Разныце пути для картинок
for ax in axes.reshape(-1):
    random_image = random.choice(images_classes)
    img = mpimg.imread(random_image[0])
    ax.set_title(random_image[1])
    ax.imshow(img)


# In[5]:


def print_loss(result): #График Loss
    plt.figure(figsize=(15,10))
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.show()


# In[6]:


IMG_SHAPE = (256,256, 3)

base_model = keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False


# In[7]:


model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(11, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()


# In[8]:


history = model.fit(train_ds, 
                    validation_data=val_ds, 
                    epochs=50,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)])
results = model.evaluate(val_ds)


# In[9]:


test_results = {}
    
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[10]:


print_loss(history)


# In[12]:


model.save(r'C:\НЕ УДАЛЯТЬ НИКОГДА НИ ПРИ КАКИХ УСЛОВИЯХ !!!!\first\first_ResNet50V2_model',save_format="h5f")


# In[ ]:




