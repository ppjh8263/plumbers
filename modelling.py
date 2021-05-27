from urllib import request
from xml.etree.ElementTree import fromstring
import numpy as np
import pandas as pd
from PIL import Image

import util

import os, glob, sys, time
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
        
imgSize=512
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
base_model = ResNet50(input_shape=(imgSize,imgSize,3), weights='imagenet', include_top=False)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=False
) 
opt=tf.keras.optimizers.Adam(
        learning_rate=lr_schedule)

loss_name='categorical_crossentropy'

df_data=util.open_pickle("./data_for_gen.pickle")
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.3,

)

tr_gen = datagen.flow_from_dataframe(df_data,
                                     x_col = 'image_path',
                                     y_col = 'name',
                                     target_size = (imgSize,imgSize),
                                     class_mode = 'categorical',
                                     batch_size = 32,
                                     interpolation = 'nearest',
                                     subset = 'training',
                                     seed = 21)
val_gen = datagen.flow_from_dataframe(df_data,
                                     x_col = 'image_path',
                                     y_col = 'name',
                                     target_size = (imgSize,imgSize),
                                     class_mode= 'categorical',
                                     batch_size= 32,
                                     interpolation= 'nearest',
                                     subset='validation',
                                     seed = 21,)


output_number=len(tr_gen.class_indices)

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(output_number, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


for layer in model.layers:
    layer.trainable = True    
    

model.compile(optimizer=opt, 
              loss=loss_name, 
              metrics=['acc'])


gpuNum= '/device:GPU:' + '0'


start = time.time()  

model_epoch=50
with tf.device(gpuNum): 
    history = model.fit(tr_gen,
                        epochs=model_epoch,
                        validation_data=val_gen)
    
trainnig_time=time.time() - start
print(trainnig_time)

result_file_name="first_model_resnet"
model_name=result_file_name+'.json'
model_weight=result_file_name+'.h5'
model_history_name=result_file_name+'.pickle'

util.model_save(model,history,model_name,model_weight,model_history_name)