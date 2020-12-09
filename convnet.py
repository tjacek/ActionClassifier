import numpy as np
import os.path
import keras,keras.backend as K,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
from keras.models import load_model
import files,spline,seqs,data

def get_train(nn_type="wide"):
    if(nn_type=="narrow"):
        read=narrow_read
        return TrainNN(read,make_narrow1D),Extract(read)
    read=seqs.read_seqs
    return TrainNN(read,basic_model),Extract(read)

def get_dataset(seqs):
    X,y=seqs.to_dataset()
    y=keras.utils.to_categorical(y)
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],
                'n_cats':y.shape[1]}
    return X,y,params

def clf_model(params):
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

def basic_exp(in_path,n_epochs=1000):
    path,name=os.path.split(in_path)
    basic_path="%s/basic" % path
    files.make_dir(basic_path)
    paths=files.get_paths(basic_path,["spline","nn","feats"])
    paths["seqs"]=in_path
    train_nn=TrainNN()
    spline.upsample(paths['seqs'],paths['spline'],size=64)
    train_nn(paths["spline"],paths["nn"],n_epochs)
    extract(paths["spline"],paths["nn"],paths["feats"])

def binary_exp(in_path,n_epochs=1000,dir_name="narrow"):
    binary_path="%s/%s" % (in_path,dir_name)
    files.make_dir(binary_path)
    input_paths=files.top_files("%s/seqs" % in_path)
    train,extract=convnet.get_train("narrow")
    ensemble1D(input_paths,binary_path,train,extract)
#   binary(input_paths,binary_path)

def ensemble1D(input_paths,out_path,train,extract,n_epochs=1000,size=64):
    funcs=[ [spline.upsample,["seqs","spline","size"]],
            [train,["spline","nn","n_epochs"]],
            [extract,["spline","nn","feats"]]]
    dir_names=["spline","nn","feats"]
    ens=EnsTransform(funcs,dir_names)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ens(input_paths,out_path, arg_dict)

if __name__ == "__main__":
    basic_exp("Data/MSR/common/seqs")