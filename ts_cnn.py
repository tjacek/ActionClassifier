import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import os.path
import keras,keras.backend as K,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
from keras.models import load_model
import files,spline,data.seqs,utils,ens,sim

def basic_exp(in_path,n_epochs=1000):
    path,name=os.path.split(in_path)
    basic_path="%s/basic" % path
    files.make_dir(basic_path)
    paths=files.get_paths(basic_path,["spline","nn","feats"])
    paths["seqs"]=in_path
    print(paths)
#    train_nn=utils.TrainNN(data.seqs.read_seqs,make_wide1D,to_dataset)
    train=get_train(nn_type="wide")
    extract=get_extract()#utils.Extract(data.seqs.read_seqs)
    spline.upsample(paths['seqs'],paths['spline'],size=64)
    train_nn(paths["spline"],paths["nn"],n_epochs)
    extract(paths["spline"],paths["nn"],paths["feats"])

def binary_exp(in_path,dir_path,n_epochs=1000):
    files.make_dir(dir_path)
    input_paths=files.top_files("%s/seqs" % in_path)
    train=get_train(nn_type="wide")
    print(input_paths)
    ensemble1D(input_paths,dir_path,train,n_epochs)

def get_train(nn_type="wide"):
    read=data.seqs.read_seqs
    if(nn_type=="narrow"):
        return utils.TrainNN(read,make_narrow1D,to_dataset)
    return utils.TrainNN(read,make_wide1D,to_dataset)

def get_extract():
    return utils.Extract(data.seqs.read_seqs)

def ensemble1D(input_paths,out_path,train,n_epochs=1000,size=64):
#    train=utils.TrainNN(data.seqs.read_seqs,make_wide1D,to_dataset)
    extract=get_extract()#utils.Extract(data.seqs.read_seqs)
    funcs=[ [spline.upsample,["seqs","spline","size"]],
            [train,["spline","nn","n_epochs"]],
            [extract,["spline","nn","feats"]]]
    dir_names=["spline","nn","feats"]
    ensemble=ens.EnsTransform(funcs,dir_names)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def to_dataset(seqs):
    X,y=seqs.to_dataset()
    n_cats=max(y)+1
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],'n_cats':n_cats}
    return X,y,params

def make_wide1D(params):
    x,input_img=basic_model(params)
    x=Dropout(0.5)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model

def basic_model(params):
    activ='relu'
    input_img=Input(shape=(params['ts_len'], params['n_feats']))
    n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
    x=Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=activ,name='conv1')(input_img)
    x=MaxPooling1D(pool_size=pool_size[0],name='pool1')(x)
    x=Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=activ,name='conv2')(x)
    x=MaxPooling1D(pool_size=pool_size[1],name='pool2')(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    return x,input_img

def make_narrow1D(params):
    input_img=Input(shape=(params['ts_len'], params['n_feats'],1))
    x=Conv2D(32,kernel_size=(8, 1),activation='relu')(input_img)
    x=MaxPooling2D(pool_size=(4, 1))(x)
    x=Conv2D(32, kernel_size=(8, 1),activation='relu')(x)
    x=MaxPooling2D(pool_size=(4, 1))(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model
    
def narrow_read(in_path):
    seq_dict=seqs.read_seqs(in_path)
    seq_dict={name_i:np.expand_dims(seq_i,axis=-1)
        for name_i,seq_i in seq_dict.items()}
    return seqs.Seqs(seq_dict)

if __name__ == "__main__":
#    ensemble1D("../clean3/base/ens/seqs","../clean3/base/ens/basic")
    binary_exp("../clean3/base/ens","../clean3/base/ens/basic")