import keras,keras.backend as K,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
from keras.models import load_model

def old_cnn(params):
    input_img=Input(shape=(64,64,1))
    x=Conv2D(32,kernel_size=(3,3),activation='relu')(input_img)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Conv2D(16, kernel_size=(3, 3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Conv2D(16, kernel_size=(3, 3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dropout(0.5)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model