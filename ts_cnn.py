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
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import load_model
import files,spline,data.seqs,utils,ens,sim

class TS_CNN(object):
    def __init__(self,nn_type="wide",l1=0.01,dropout=0.5,
                activ='relu',optim_alg=None):
        if(optim_alg is None):
            optim_alg=Nestrov()
        self.nn_type=nn_type
        self.activ=activ
        self.l1=l1
        self.dropout=dropout
        self.optim_alg=optim_alg

    def __call__(self,params):
        input_img=Input(shape=(params['ts_len'], params['n_feats']))
        if(self.nn_type=="narrow"):
            n_kerns,kern_size,pool_size=[32,32],[(8,1),(8,1)],[(4,1),(4,1)]
            x=add_conv_layer(input_img,n_kerns,kern_size,
                            pool_size,activ=self.activ,one_dim=False)
        else:
            n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
            x=add_conv_layer(input_img,n_kerns,kern_size,
                            pool_size,activ=self.activ,one_dim=True)
        x=Flatten()(x)
        reg=regularizers.l1(self.l1) if(self.l1) else None
        x=Dense(100, activation=self.activ,name="hidden",kernel_regularizer=reg)(x)
        x=self.reg_layer(x)
        x=Dense(units=params['n_cats'],activation='softmax')(x)
        model = Model(input_img, x)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=self.optim_alg())
        model.summary()
        return model

    def reg_layer(self,x):
        if(self.dropout=="batch_norm"):
            return BatchNormalization()(x)
        if(self.dropout):
            return Dropout(self.dropout)(x)
        return x

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

def simple_exp(in_path,n_epochs=1000):
    print(paths)
    train,extract=get_train(nn_type="wide")
    utils.single_exp_template(in_path,out_name,train,extract,seq_path)

def ensemble_exp(in_path,out_name,n_epochs=1000,size=64):
    input_paths=files.top_files("%s/seqs" % in_path)
    out_path="%s/%s" % (in_path,out_name)
    files.make_dir(out_path)
    train,extract=get_train(nn_type="wide")
    ensemble=ens.ts_ensemble(train,extract)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def get_train(nn_type="wide"):
    read=data.seqs.read_seqs
    train=utils.TrainNN(read,TS_CNN(optim_alg=RMS()),to_dataset)
    extract=utils.Extract(read)
    return train,extract

def to_dataset(seqs):
    X,y=seqs.to_dataset()
    n_cats=max(y)+1
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],'n_cats':n_cats}
    return X,y,params
    
def narrow_read(in_path):
    seq_dict=seqs.read_seqs(in_path)
    seq_dict={name_i:np.expand_dims(seq_i,axis=-1)
        for name_i,seq_i in seq_dict.items()}
    return seqs.Seqs(seq_dict)

if __name__ == "__main__":
    ensemble_exp("../dtw_paper/MHAD/binary/","1D_CNN_rms",n_epochs=1000)
#    binary_exp("../dtw_paper/MHAD/binary/","../dtw_paper/MHAD/binary/1D_CNN_128")