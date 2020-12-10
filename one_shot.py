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

def dtw_one_shot(in_path,out_path=None,n_epochs=5):
    dtw_feats=feats.read_feats(in_path)
    train,test=dtw_feats.split()
    X,y=to_sim_dataset(train)
    params={'input_shape':(dtw_feats.dim(),)}
    siamese_net,extractor=sim.build_siamese(params,get_large_model())
    siamese_net.fit(X,y,epochs=n_epochs,batch_size=64)
    if(out_path):
        extractor.save(out_path)

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

def to_sim_dataset(dtw_feats):
    dtw_feats.norm()
    pairs=sim.all_pairs(dtw_feats.names())
    X,y=[],[]
    for name_i,name_j in pairs:
        X.append((dtw_feats[name_i],dtw_feats[name_j]))
        y.append(int(name_i.split('_')[0]==name_j.split('_')[0]))
    X=np.array(X)
    X=[X[:,0],X[:,1]]
#    y=keras.utils.to_categorical(y)
    return X,y

if __name__ == "__main__":
    in_path=["dtw/maxz/feats","dtw/corl/feats"]
    dtw_one_shot(in_path,"dtw_nn",300)
    dtw_extract(in_path,"dtw_nn","sim_feats")