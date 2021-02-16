from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
import data.seqs,files,utils,spline,ens
import sim,ts_cnn

def binary_exp(in_path,dir_path,n_epochs=1000):
    files.make_dir(dir_path)
    input_paths=files.top_files("%s/seqs" % in_path)
    train=get_train()
    print(input_paths)
    ensemble1D(input_paths,dir_path,train,n_epochs)

def ensemble1D(input_paths,out_path,train,n_epochs=1000,size=64):
    extract= utils.Extract(data.seqs.read_seqs)
    funcs=[ [spline.upsample,["seqs","spline","size"]],
            [train,["spline","nn","n_epochs"]],
            [extract,["spline","nn","feats"]]]
    dir_names=["spline","nn","feats"]
    ensemble=ens.EnsTransform(funcs,dir_names)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def sim_ts(model,params):
    activ='relu'
    n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
    n_hidden=100
    model.add(Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=activ,name='conv1'))
    model.add(MaxPooling1D(pool_size=pool_size[0],name='pool1'))
    model.add(Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=activ,name='conv2'))
    model.add(MaxPooling1D(pool_size=pool_size[1],name='pool2'))
    model.add(Flatten())
#    model.add(Dropout(0.5))
    model.add(Dense(n_hidden, activation=activ,name='hidden',kernel_regularizer=regularizers.l1(0.01)))
    return model

def get_train():
	get_model=sim_ts
	get_cat=sim.all_cat
	read=data.seqs.read_seqs
	params={'input_shape':(64,100)}
	train_sim=sim.SimTrain(get_model,get_cat,read,4)
	def train(in_path,out_path,n_epochs):
		return train_sim(in_path,out_path,n_epochs,params=params)
	return train

in_path="../clean3/agum/ens"
out_path="../clean3/agum/ens/sim"
#out_path="test"
#train=get_train()
#train(in_path,out_path,n_epochs=5)
binary_exp(in_path,out_path,n_epochs=400)