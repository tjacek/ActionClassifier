import keras,keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np
import re,cv2
import data
from keras import backend as K
import os
import importlib

def extract_frame_feats(in_path,nn_path,out_path):
    model=load_model(nn_path)
    action_dirs=data.top_files(in_path)
    action_dirs.sort(key=data.natural_keys)
    X,y=[],[]
    data.make_dir(out_path)
    for action_i in action_dirs:
        frame_path=data.top_files(action_i)
        frame_path.sort(key=data.natural_keys)
        def frame_helper(path_ij):
            frame_ij=read_frame(path_ij)
            frame_ij=np.expand_dims(frame_ij,0)
            return model.predict(frame_ij)
        pred_i=np.array([frame_helper(path_ij) for path_ij in frame_path])
        out_i=out_path+'/'+action_i.split('/')[-1]
        pred_i=np.squeeze(pred_i, axis=1)
        np.savetxt(out_i, pred_i, delimiter=',' )

def train_model(in_path,out_path,n_epochs=10):
    train_X,train_y=read_imgs(in_path)
    model=make_conv()
    model.summary()
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    model.save(out_path)

def read_imgs(in_path,n_split=4):
    action_dirs=data.top_files(in_path)
    action_dirs.sort(key=data.natural_keys)
    X,y=[],[]
    for action_i in action_dirs:
        person_i= get_person(action_i)
        if((person_i%2)==1):
            frame_path=data.top_files(action_i)
            frame_path.sort(key=data.natural_keys)
            for frame_ij_path in frame_path:
                X.append(read_frame(frame_ij_path))                
                y.append( person_i/2)
    return np.array(X),keras.utils.to_categorical(y)

def read_frame(frame_ij_path,n_split=4):
    frame_ij=cv2.imread(frame_ij_path,0)
    return np.array(np.vsplit(frame_ij,n_split)).T

def get_person(action_i):
    name_i=action_i.split("/")[-1]
    name_i=re.sub('[a-z]','',name_i)
    return int(name_i.split('_')[1])

def make_conv():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',name="hidden"))
    model.add(Dropout(0.25))
    model.add(Dense(units=5, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True)
              optimizer=keras.optimizers.Adadelta())
    return model

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

#train_model("mra","persons",n_epochs=100)
extract_frame_feats("mra","persons","person_feats")