from tcn import TCN, tcn_full_summary
import numpy as np
import keras.losses
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.models import Model
import data.feats,utils

def make_tcn(params):
    tcn_layer = TCN(input_shape=(params["ts_len"], params["n_feats"]),name="tcn_layer")
    m = Sequential([tcn_layer,Dense(units=params['n_cats'],activation='softmax')])
    m.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy)
    tcn_full_summary(m, expand_residual_blocks=False)
    return m

def simple_exp(in_path,out_path,n_epochs=10):
    train,extract=train_tcn,extract_tcn
    utils.single_exp_template(in_path,out_path,train,extract,
        n_epochs=n_epochs)

def train_tcn(in_path,nn_path,n_epochs=5):
    dataset=data.seqs.read_seqs(in_path)
    train,test=dataset.split()
    X,y,params=to_dataset(train)
    if("n_cats" in params ):
        y=utils.to_one_hot(y,params["n_cats"])
    model=make_tcn(params)
    model.fit(X,y,epochs=n_epochs,batch_size=32)
    model.save_weights(nn_path)

def extract_tcn(in_path,nn_path,out_path):
    dataset=data.seqs.read_seqs(in_path)
    X,y,params=to_dataset(dataset)
    model=make_tcn(params)
    model.load_weights(nn_path)    
    print(dir(model.input))
    extractor=tf.keras.models.Model(inputs=model.input,
                outputs=model.get_layer("tcn_layer").output)
    extractor.summary()
    feats=data.feats.Feats()
    for i,name_i in enumerate(dataset.names()):
        x_i=np.array(dataset[name_i])
        x_i=np.expand_dims(x_i,axis=0)
        feats[name_i]= extractor.predict(x_i)
#        raise Exception(feat[name_i].shape)
    feats.save(out_path)

def to_dataset(seqs):
    X,y=seqs.to_dataset()
    n_cats=max(y)+1
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],'n_cats':n_cats}
    return X,y,params


in_path="../3DHOI/spline"
simple_exp(in_path,"test")