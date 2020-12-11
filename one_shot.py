import keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,Lambda#GlobalAveragePooling1D
from keras import regularizers
from keras.models import load_model
import numpy as np
import feats,sim

class DtwModel(object):
    def __init__(self,n_feats):
        self.n_feats=n_feats

    def __call__(self,model):
        activ='relu'
        for n_feats_i in self.n_feats[:-1]:
            model.add(Dense(n_feats_i, activation=activ))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_feats[-1], activation=activ,name='hidden',
            kernel_regularizer=regularizers.l1(0.01)))
        return model

def get_basic_model():
    return DtwModel([100,64])

def get_large_model():
    return DtwModel([128,96,64])

def early_drop(model):
    model.add(Dropout(0.75))
    model.add(Dense(196, activation='relu'))
#    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu',name='hidden',
            kernel_regularizer=regularizers.l1(0.01)))
    return model

def dtw_one_shot(in_path,out_path=None,n_epochs=5):
    dtw_feats=feats.read_feats(in_path)
    dtw_feats.norm()
    def all_cat(name_i,name_j):
        return int(name_i.split('_')[0]==name_j.split('_')[0])
    make_model=get_basic_model()
    sim_train=sim.SimTrain(make_model,all_cat)
    sim_train(dtw_feats,out_path,n_epochs)

def dtw_extract(in_path,nn_path,out_path):
    dtw_feats=feats.read_feats(in_path)
    dtw_feats.norm()
    extractor=load_model(nn_path)
    def helper(x_i):
        x_i=np.expand_dims(x_i,0)   
        result_i= extractor.predict(x_i)
        return result_i
    dtw_feats=dtw_feats.transform(helper)
    dtw_feats.save(out_path)

if __name__ == "__main__":
#    in_path=["dtw/maxz/feats","dtw/corl/feats"]
    in_path=['../agum/max_z/dtw','../agum/corl/dtw','../agum/skew/dtw']
    in_path=['../agum/max_z/person','../agum/corl/person','../agum/skew/person']
    dtw_one_shot(in_path,"dtw_nn",300)
    dtw_extract(in_path,"dtw_nn","sim_feats")