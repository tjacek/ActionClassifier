import numpy as np
import os,os.path
import lstm

def train_binary_models(in_path,out_path,n_epochs=10):
    train_X,train_y,test_X,test_y,params=lstm.prepare_data(in_path)
    all_cats=params['n_cats']
    params['n_cats']=2    
    make_dir(out_path)
    for cat_i in range(all_cats):
        print("model category %d" % cat_i)
        binary_y=binarize(train_y,cat_i)
        model_i=lstm.make_conv_lstm(params)
        model_i.fit(train_X,binary_y,epochs=n_epochs,batch_size=32)
        out_i=out_path+'/nn'+str(cat_i)
        model_i.save(out_i)

def binarize(train_y,cat_i):
    y=train_y.copy()
    if(0<cat_i and cat_i < y.shape[1]-1):
        y[:,(cat_i+1):]=0
    y[:,:cat_i]=0
    return y

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

train_binary_models("mra","binary",n_epochs=10)