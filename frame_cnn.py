import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import keras
from keras.layers import Input,Dense
from keras.models import Model
import deep

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

def train(in_path,nn_path,n_epochs=5):
    train_model=utils.TrainNN(read_proj,old_cnn,to_dataset)
    train_model(in_path,nn_path,n_epochs)

def extract(in_path,nn_path,out_path):
    extract=utils.ExtractSeqs(read_proj)
    extract(in_path,nn_path,out_path)

def read_proj(in_path):
    dataset=data.imgs.read_frame_seqs(in_path)
    def helper(frame_i):
        return np.array(np.vsplit(frame_i,3)).T
    dataset.transform(helper,new=False,single=True)
    return dataset

def to_dataset(train):
    X,y=[],[]
    for name_i,seq_i in train.items():
        cat_i=name_i.get_cat()
        for frame_j in seq_i:
            X.append(frame_j)
            y.append(cat_i)
    params={"dim":train.dims(),"n_cats":train.n_cats()}
    return np.array(X),y,params

#def old_cnn(params):
#    input_img=Input(shape=params["dim"])
#    x=Conv2D(32,kernel_size=(3,3),activation='relu')(input_img)
#    x=MaxPooling2D(pool_size=(2,2))(x)
#    x=Conv2D(16, kernel_size=(3, 3),activation='relu')(x)
#    x=MaxPooling2D(pool_size=(2, 2))(x)
#    x=Conv2D(16, kernel_size=(3, 3),activation='relu')(x)
#    x=MaxPooling2D(pool_size=(2, 2))(x)
#    x=Flatten()(x)
#    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
#    x=Dropout(0.5)(x)
#    x=Dense(units=params['n_cats'],activation='softmax')(x)
#    model = Model(input_img, x)
#    model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
#    model.summary()
#    return model

def ens_exp(in_path,out_path,n_epochs=5,n_cats=12):
    binary_train=binary_gen(in_path,n_epochs)
    funcs=[[extract,["in_path","nn","feats"]]]
    dir_names=["feats"]
    arg_dict={'in_path':in_path}        
    binary_ens=ens.BinaryEns(binary_train,funcs,dir_names)
    binary_ens(out_path,n_cats,arg_dict)

def binary_gen(in_path,n_epochs=5):
    dataset=read_proj(in_path)
    train,test=dataset.split()
    X,y,params=to_dataset(train)
    params["n_cats"]=2
    def binary_train(nn_path,i):
        y_i=ens.binarize(y,i)
        model=old_cnn(params)
        model.fit(X,y_i,epochs=n_epochs,batch_size=32)
        model.save(nn_path)
    return binary_train

frame_cnn=FrameCNN()
frame_cnn({'dims':(64,64,1),'n_cats':20})