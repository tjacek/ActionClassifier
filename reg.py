import numpy as np
from keras.models import Model
from keras.layers import Input,Dense
import deep,data.seqs,data.feats,stats,utils

class TS_REG(object):
    def __init__(self,optim_alg=None,activ='relu'):
        if(optim_alg is None):
            optim_alg=deep.RMS()
        self.optim_alg=optim_alg
        self.activ=activ

    def __call__(self,params):
        input_img=Input(shape=(params['ts_len'], params['n_feats']))
        n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
        x=deep.add_conv_layer(input_img,n_kerns,kern_size,
                            pool_size,activ=self.activ,one_dim=True)
        x=deep.full_layer(x,l1=None)
        x=Dense(units=params['n_outputs'],activation='relu')(x)
        model = Model(input_img, x)
        model.compile(loss='mean_squared_error',
              optimizer=self.optim_alg())
        model.summary()
        return model

def simple_exp(in_path,out_path,n_epochs=1000):
    read=data.seqs.read_seqs
    train=utils.TrainNN(read,TS_REG(),to_dataset)
    extract=utils.Extract(read)
    utils.single_exp_template(in_path,out_path,
    	train,extract,size=64,n_epochs=n_epochs)

def to_dataset(seqs):
    get_stats=stats.get_base_stats()
    stat_feats=get_stats.compute_feats(seqs)
    X=seqs.to_dataset()[0]
    y=stat_feats.to_dataset()[0]
    params={'ts_len':X[0].shape[0],'n_feats':X[0].shape[1],
    		'n_outputs':y.shape[1]}
    return X,y,params

def check_regression(seq_path,nn_path):
    read=data.seqs.read_seqs
    extract=utils.Extract(read,name=None)
    reg_feats=extract(seq_path,nn_path,None)
    seq_dict=data.seqs.read_seqs(seq_path)
    get_stats=stats.get_base_stats()
    stat_feats=get_stats.compute_feats(seq_dict)
    reg_feats=reg_feats.split()[0]
    stat_feats=stat_feats.split()[0]
    res=[ #np.linalg.norm(reg_feats[name_i]-stat_feats[name_i]) 
    		np.mean(np.abs(reg_feats[name_i]-stat_feats[name_i]))
    		for name_i in reg_feats.keys()]
    print(res)
#    feat_dict=data.feats.read(feat_path)
#    print(len(feat_dict))

seq_path="../dtw_paper/MSR/binary/seqs/nn0"
simple_exp(seq_path,"test",n_epochs=1000)
#spline_path="test/spline"
#nn_path="test/nn"
#check_regression(spline_path,nn_path)