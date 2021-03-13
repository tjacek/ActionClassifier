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

class TS_CNN(object):
    def __init__(self,nn_type,activ='relu'):
        self.nn_type=nn_type
        self.activ=activ

    def __call__(self,params):
#        activ='relu'
        input_img=Input(shape=(params['ts_len'], params['n_feats']))
        n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
        if(self.nn_type=="narrow"):
            x=Conv2D(32,kernel_size=(8, 1),activation=self.activ)(input_img)
            x=MaxPooling2D(pool_size=(4, 1))(x)
            x=Conv2D(32, kernel_size=(8, 1),activation=self.activ)(x)
            x=MaxPooling2D(pool_size=(4, 1))(x)
        else:
            x=Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=self.activ,name='conv1')(input_img)
            x=MaxPooling1D(pool_size=pool_size[0],name='pool1')(x)
            x=Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=self.activ,name='conv2')(x)
            x=MaxPooling1D(pool_size=pool_size[1],name='pool2')(x)
        x=Flatten()(x)
        x=Dense(100, activation=self.activ,name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
        x=Dropout(0.5)(x)
        x=Dense(units=params['n_cats'],activation='softmax')(x)
        model = Model(input_img, x)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
        model.summary()
        return model

def simple_exp(in_path,n_epochs=1000):
    print(paths)
    train=get_train(nn_type="wide")
    extract=utils.Extract(data.seqs.read_seqs)
    utils.single_exp_template(in_path,out_name,train,extract,seq_path)

def ensemble_exp(in_path,out_name,n_epochs=1000,size=64):
    input_paths=files.top_files("%s/seqs" % in_path)
    out_path="%s/%s" % (in_path,out_name)
    files.make_dir(out_path)
    train=get_train(nn_type="wide")
    extract=utils.Extract(data.seqs.read_seqs)
    ensemble=ens.ts_ensemble(train,extract)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def get_train(nn_type="wide"):
    read=data.seqs.read_seqs
    return utils.TrainNN(read,TS_CNN(nn_type,"tanh"),to_dataset)

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
    ensemble_exp("../dtw_paper/MSR/binary","1D_CNN_tanh",n_epochs=1000)
#    binary_exp("../dtw_paper/MHAD/binary/","../dtw_paper/MHAD/binary/1D_CNN_128")