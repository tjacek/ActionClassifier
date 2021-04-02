import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import keras
from keras.layers import Input,Dense
from keras.models import Model
import deep,utils,files,ens,data.imgs

class FrameCNN(object):
    def __init__(self,dropout="batch_norm",l1=0.01,
                        activ='relu',optim_alg=None):
        if(optim_alg is None):
            optim_alg=deep.RMS()
        self.dropout=dropout
        self.activ=activ
        self.l1=l1
        self.optim_alg=optim_alg

    def __call__(self,params):
        input_img=Input(shape=(params['dims']))
        n_kerns,kern_size=[32,16,16],[(3,3),(3,3),(3,3)]
        pool_size=[(2,2),(2,2),(2,2)]
        x=deep.add_conv_layer(input_img,n_kerns,kern_size,
                            pool_size,activ=self.activ,one_dim=False)
        x=deep.full_layer(x,size=100,l1=self.l1,
            dropout=self.dropout,activ=self.activ)
        x=Dense(units=params['n_cats'],activation='softmax')(x)
        model = Model(input_img, x)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=self.optim_alg())
        model.summary()
        return model

def simple_exp(in_path,out_path,n_epochs=1000):
    paths=files.prepare_dirs(out_path,["nn","feats"])
    paths['frames']=in_path
    train,extract=get_train()
    train(paths["frames"],paths["nn"],n_epochs=n_epochs)
    extract(paths["frames"],paths["nn"],paths["feats"])

def get_train():
    frame_cnn=FrameCNN()
    train=utils.TrainNN(read_proj,frame_cnn,to_dataset)
    extract=utils.ExtractSeqs(read_proj)
    return train,extract

def read_proj(in_path,n_split=None):
    dataset=data.imgs.read_frame_seqs(in_path)
    if(n_split is None):
        n_split=get_n_split(dataset)
    def helper(frame_i):
        return np.array(np.vsplit(frame_i,n_split)).T
    dataset.transform(helper,new=False,single=True)
    return dataset

def get_n_split(dataset):
    dims=dataset.dims()
    max_i,min_i=max(dims),min(dims)
    return int(max_i/min_i)

def to_dataset(train):
    X,y=[],[]
    for name_i,seq_i in train.items():
        cat_i=name_i.get_cat()
        for frame_j in seq_i:
            X.append(frame_j)
            y.append(cat_i)
    params={"dims":train.dims(),"n_cats":train.n_cats()}
    return np.array(X),y,params

def ens_exp(in_path,out_path,n_epochs=5,n_cats=20):
    binary_train=binary_gen(in_path,FrameCNN(),n_epochs)
    extract=utils.ExtractSeqs(read_proj)
    funcs=[[extract,["in_path","nn","seqs"]]]
    dir_names=["seqs"]
    arg_dict={'in_path':in_path}        
    binary_ens=ens.BinaryEns(binary_train,funcs,dir_names)
    binary_ens(out_path,n_cats,arg_dict)

def binary_gen(in_path,frame_cnn,n_epochs=5):
    dataset=read_proj(in_path)
    train,test=dataset.split()
    X,y,params=to_dataset(train)
    params["n_cats"]=2
    def binary_train(nn_path,i):
        y_i=ens.binarize(y,i)
        model=frame_cnn(params)
        model.fit(X,y_i,epochs=n_epochs,batch_size=32)
        model.save(nn_path)
    return binary_train

ens_exp("../time","test2",n_epochs=5)