import numpy as np
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense,Dropout
from keras import regularizers
from keras.models import load_model
import sys

def make_model(img_shape=(64,64,1),n_dense=1):
    input_img = Input(shape=img_shape)
    x=input_img
    kern_size,pool_size,filters=(6,6),(4,4),[32,16,16,16]
    for filtr_i in filters:
        x = Conv2D(filtr_i, kern_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size, padding='same')(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dropout(0.5)(x)
    x=Dense(n_dense, activation="linear")(x)  
    model = Model(input_img, x)
    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(lr=0.00001))
    model.summary()
    return model