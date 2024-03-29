import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import os.path
import keras,keras.backend as K,keras.utils
from keras.models import Model,Sequential,load_model
from keras.layers import Input,Dense
import files,data.seqs,utils,ens,sim
import deep

class TS_CNN(object):
    def __init__(self,nn_type="wide",l1=0.01,dropout="batch_norm",
                activ='relu',optim_alg=None):
        if(optim_alg is None):
            optim_alg=deep.RMS()
        self.nn_type=nn_type
        self.activ=activ
        self.l1=l1
        self.dropout=dropout
        self.optim_alg=optim_alg

    def __call__(self,params):
        input_img=Input(shape=(params['ts_len'], params['n_feats']))
        if(self.nn_type=="narrow"):
            n_kerns,kern_size,pool_size=[32,32],[(8,1),(8,1)],[(4,1),(4,1)]
            x=deep.add_conv_layer(input_img,n_kerns,kern_size,
                            pool_size,activ=self.activ,one_dim=False)
        else:
            n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
            x=deep.add_conv_layer(input_img,n_kerns,kern_size,
                            pool_size,activ=self.activ,one_dim=True)
        x=deep.full_layer(x,100,self.l1,self.dropout,self.activ)
        x=Dense(units=params['n_cats'],activation='softmax')(x)
        model = Model(input_img, x)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=self.optim_alg())
        model.summary()
        return model

def simple_exp(in_path,out_path,n_epochs=1000):
    ts_cnn=TS_CNN(dropout=0.5,activ='elu',
                    optim_alg=deep.Nestrov())
    train,extract=get_train(ts_cnn)
    utils.single_exp_template(in_path,out_path,train,extract)

def ensemble_exp(in_path,out_name,n_epochs=1000,size=64):
    input_paths=files.top_files("%s/seqs" % in_path)
    out_path="%s/%s" % (in_path,out_name)
    ts_cnn=TS_CNN(dropout=0.5,activ='elu',
                    optim_alg=deep.Nestrov())
    train,extract=get_train(ts_cnn)
    ensemble=ens.ts_ensemble(train,extract)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def multi_exp(in_path,out_name,n_epochs=1000,size=64):
    out_path="%s/%s" % (in_path,out_name)
    train_dict={"no_l1":TS_CNN(l1=None),"dropout_0.5":TS_CNN(dropout=0.5),
        "adam":TS_CNN(optim_alg=deep.Adam()),
        "nestrov":TS_CNN(optim_alg=deep.Nestrov()),
        "tanh":TS_CNN(activ='tanh'),"base":TS_CNN()}
    train_dict={name_i:get_train(train_i)
                    for name_i,train_i in train_dict.items()}
    arg_dict={'size':size,'n_epochs':n_epochs}
    ens.multimodel_ensemble(in_path,out_path,train_dict,arg_dict)

def get_train(ts_cnn=None):
    if(ts_cnn is None):
        ts_cnn=TS_CNN()
    read=data.seqs.read_seqs
    train=utils.TrainNN(read,ts_cnn,to_dataset)
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
    ensemble_exp("../dtw_paper/MHAD/binary",'elu',n_epochs=1000)
#    binary_exp("../dtw_paper/MHAD/binary/","../dtw_paper/MHAD/binary/1D_CNN_128")