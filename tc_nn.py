from tcn import TCN, tcn_full_summary
import keras.losses
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import data.seqs,utils

def make_tcn(params):
    tcn_layer = TCN(input_shape=(params["ts_len"], params["n_feats"]))
    m = Sequential([tcn_layer,Dense(units=params['n_cats'],activation='softmax')])
    m.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy)
    tcn_full_summary(m, expand_residual_blocks=False)
    return m

def simple_exp(in_path,out_path,n_epochs=1000):
    ts_cnn=make_tcn
    read=data.seqs.read_seqs
    train=utils.TrainNN(read,ts_cnn,to_dataset)
    extract=utils.Extract(read)
    utils.single_exp_template(in_path,out_path,train,extract)

def to_dataset(seqs):
    X,y=seqs.to_dataset()
    n_cats=max(y)+1
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],'n_cats':n_cats}
    return X,y,params

in_path="../3DHOI/spline"
simple_exp(in_path,"test")