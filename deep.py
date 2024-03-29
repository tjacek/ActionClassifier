import keras
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D#,Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dropout,Flatten,Dense
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import keras.losses

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

def add_model_conv(model,n_kerns,kern_size,pool_size,activ='relu'):
    for i,n_kern_i in enumerate(n_kerns):
        conv_i,pool_i='conv%d' % i,'pool%d' % i
        model.add(Conv1D(n_kern_i, kernel_size=kern_size[i],
                        activation=activ,name=conv_i))    
        model.add(MaxPooling1D(pool_size=pool_size[i],name=pool_i))

def full_layer(x,size=100,l1=0.01,dropout=0.5,activ='relu'):
    x=Flatten()(x)
    reg=regularizers.l1(l1) if(l1) else None
    name="prebatch" if(dropout=="batch_norm") else "hidden"
    if(type(size)==list):
        for size_i in size[:-1]:
            x=Dense(size_i, activation=activ,kernel_regularizer=None)(x)
        size=size[-1]
    x=Dense(size, activation=activ,name=name,kernel_regularizer=reg)(x)
    if(dropout=="batch_norm"):
        return BatchNormalization(name="hidden")(x)
    if(dropout):
        return Dropout(dropout)(x)

def clf_nn(x,model_input,n_cats,optim_alg):
    x=Dense(units=n_cats,activation='softmax')(x)
    clf_model = Model(model_input, x)
    clf_model.summary()
    clf_model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=optim_alg())
    return clf_model

def lstm_cnn(model,n_kern,kern_size,pool_size,activ,input_shape):
    for i,n_kern_i in enumerate(n_kern):
        if(i==0):
            conv_i=Conv2D(n_kern_i,kern_size[i], padding='same')
            model.add(TimeDistributed(conv_i,input_shape=input_shape))
        else:
            conv_i=Conv2D(n_kern_i,kern_size[i])
            model.add(TimeDistributed(conv_i))
        model.add(TimeDistributed(Activation(activ)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size[i])))