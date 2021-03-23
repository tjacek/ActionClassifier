import keras
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D#,Lambda
from keras.layers import Dropout,Flatten,Dense
from keras.layers.normalization import BatchNormalization


class Nestrov(object):
    def __init__(self,lr=0.001,momentum=0.9):
        self.lr=lr
        self.momentum=momentum

    def __call__(self):
        return keras.optimizers.SGD(lr=self.lr,  momentum=self.momentum, nesterov=True)

class Adam(object):
    def __init__(self,lr=0.001):
        self.lr=lr
    def __call__(self):
        return keras.optimizers.Adam(learning_rate=self.lr)

class RMS(object):
    def __init__(self,lr=0.001):
        self.lr=lr
    def __call__(self):
        return keras.optimizers.RMSprop(learning_rate=self.lr)

def add_conv_layer(input_img,n_kerns,kern_size,
                    pool_size,activ='relu',one_dim=False):
    x=input_img
    Conv=Conv1D if(one_dim) else Conv2D
    MaxPooling=MaxPooling1D if(one_dim) else MaxPooling2D
    for i,n_kern_i in enumerate(n_kerns):
        x=Conv(n_kern_i, kernel_size=kern_size[i],activation=activ,name='conv%d'%i)(x)
        x=MaxPooling(pool_size=pool_size[i],name='pool%d' % i)(x)
    return x

def full_layer(x,l1=0.01,dropout=0.5,activ='relu'):
    x=Flatten()(x)
    reg=regularizers.l1(l1) if(l1) else None
    name="prebatch" if(dropout=="batch_norm") else "hidden"
    x=Dense(100, activation=activ,name=name,kernel_regularizer=reg)(x)
    if(dropout=="batch_norm"):
        return BatchNormalization(name="hidden")(x)
    if(dropout):
        return Dropout(dropout)(x)