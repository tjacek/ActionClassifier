import keras,keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np
import re,cv2
import data,binary
from keras import backend as K
import os
import importlib
import random
from keras import regularizers

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

def train_binary_model(in_path,out_path,n_epochs=10):
    train_X,train_y=read_imgs(in_path)
    data.make_dir(out_path)
    n_cats=train_y.shape[1]
    for cat_i in range(n_cats):
        y_i=binary.binarize(train_y,cat_i)
        model=make_conv(2)
        model.summary()
        model.fit(train_X,y_i,epochs=n_epochs,batch_size=1000)
        out_i=out_path+'/nn'+str(cat_i)
        model.save(out_i)

def train_model(in_path,out_path,n_epochs=10):
    train_X,train_y=read_imgs(in_path)
    model=make_conv(train_y.shape[1])
    model.summary()
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    model.save(out_path)

def read_imgs(in_path,n_split=4):
    action_dirs=data.top_files(in_path)
    #action_dirs.sort(key=data.natural_keys)
    random.shuffle(action_dirs)
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

def make_conv(n_cats):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),))
    model.add(Dropout(0.5))
    model.add(Dense(units=n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
              optimizer=keras.optimizers.Adadelta())
    return model

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

#train_model("frames","persons",n_epochs=100)
train_binary_model("frames","binary_persons",n_epochs=100)
#extract_frame_feats("mra","persons","person_feats")