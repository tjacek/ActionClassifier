import keras,keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from scipy.interpolate import CubicSpline
import data,utils

def simple_exp(in_path,n_epochs=100):
    utils.simple_exp(in_path,n_epochs=n_epochs,
                        prepare=prepare_data,make_model=make_conv)

def prepare_data(in_path,n=128):
    (train_X,train_y),(test_X,test_y)=data.get_dataset(in_path)
    params={'ts_len':n,
            'n_feats':train_X[0].shape[1],
            'n_cats':np.unique(train_y).shape[0]}    
    train_X,train_y=format_data(train_X,train_y) 
    test_X,test_y=format_data(test_X,test_y) 
    return train_X,train_y,test_X,test_y,params

def format_data(X,y,new_size=128):
    new_X,new_y=[],keras.utils.to_categorical(y)
    new_range=np.arange(new_size) 
    for i,x_i in enumerate(X):
    	old_size=x_i.shape[0]
        old_range=np.arange(old_size).astype(float)
        step=float(new_size)/float(old_size)
        old_range*=step
        new_X.append([])
        for feat_j in x_i.T:
            cs_ij=CubicSpline(old_range,feat_j)
            x_i=cs_ij(new_range)
            new_X[-1].append(x_i)
    new_X= np.swapaxes(np.array(new_X),1,2)
    new_X= np.expand_dims(new_X,-1)
    return new_X,new_y

def make_conv(params):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(8, 1),
                 activation='relu',
                 input_shape=(params['ts_len'],params['n_feats'],1)))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(8, kernel_size=(8, 1),
                 activation='relu',
                 input_shape=(params['ts_len'],params['n_feats'],1)))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(units=params['n_cats'], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              optimizer=keras.optimizers.SGD(lr=0.01,  momentum=0.9, nesterov=True))
    #          metrics=['accuracy'])
    return model

simple_exp("mra")