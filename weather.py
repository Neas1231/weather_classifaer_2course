#!/usr/bin/env python
# coding: utf-8
import os
import PIL as pl
from PIL import Image
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from st_on_hover_tabs import on_hover_tabs

dictuar = {0:'dew' ,1:'fogsmog' ,2:'frost' ,3:'glaze' ,4:'hail' ,5:'lightning' , 6:'rain', 7:'rainbow', 
           8:'rime', 9:'sandstorm', 10:'snow'}

st.set_page_config(layout="wide")
st.header("Программа для классификации погодных условий ")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

ResNet_model = tf.keras.models.load_model(r'first_ResNet50V2_model',compile=False)
ResNet_model.compile()

uploaded_file = st.file_uploader('Загрузите картинку погоды, которую попробует распознать нейросеть!')
if uploaded_file is not None:

       bytes_data = uploaded_file.getvalue()

       image = Image.open(uploaded_file)
       image.save(rf'{os.getcwd()}/images/img/IMG.jpg')
       image_copy = image

       datagen= keras.preprocessing.image.ImageDataGenerator(
       validation_split=0.25, 
       rescale=1./255 
       )
    
       image = datagen.flow_from_directory(
       f'{os.getcwd()}',
       target_size=(256, 256), 
       batch_size=32,
       class_mode='categorical'
       )

       preds = ResNet_model.predict(image)

       if st.button('Показать результат'):
            st.write(dictuar[preds.argmax()])

       with st.sidebar:
               tabs = on_hover_tabs(tabName=['Предсказание', 'Картинка', 'Все вместе'], 
               iconName=['dashboard', 'money', 'economy'],
               styles = {'navtab': {'background-color':'#111',
                                           'color': '#818181',
                                           'font-size': '18px',
                                           'transition': '.3s',
                                           'white-space': 'nowrap',
                                           'text-transform': 'uppercase'},
                         'iconStyle':{'position':'fixed',
                                           'left':'7.5px',
                                           'text-align': 'left'},
                         'tabStyle' : {'list-style-type': 'none',
                                            'margin-bottom': '30px',
                                            'padding-left': '30px'}},
                                 key="1")
                         

       if tabs =='Предсказание':
           st.header(dictuar[preds.argmax()])         
       if  tabs =='Картинка':
           st.image(image_copy)
       if tabs == 'Все вместе':
           st.header(dictuar[preds.argmax()]) 
           st.image(image_copy)
        
   
    
   
    
    
        
    
