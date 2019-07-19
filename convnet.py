import keras,keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from scipy.interpolate import CubicSpline
import data

def prepare_data(in_path,n=128):
    (train_X,train_y),(test_X,test_y)=data.get_dataset(in_path)
    train_X,train_y=format_data(train_X,train_y) 
    test_X,test_y=format_data(train_X,train_y) 

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
    return new_X,new_y

def make_conv(params):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

prepare_data("mra")