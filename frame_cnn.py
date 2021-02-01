import numpy as np
import keras,keras.backend as K,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
import utils,data.imgs,ens

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

def old_cnn(params):
    input_img=Input(shape=params["dim"])
    x=Conv2D(32,kernel_size=(3,3),activation='relu')(input_img)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Conv2D(16, kernel_size=(3, 3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Conv2D(16, kernel_size=(3, 3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dropout(0.5)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model

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
        model.fit(X,y_i,epochs=n_epochs)
        model.save(nn_path)
    return binary_train

#train("../3DHOI","../nn_test",n_epochs=5)
#extract("../3DHOI","../nn_test","../nn_seqs")
ens_exp("../3DHOI","../ens",n_epochs=5,n_cats=12)