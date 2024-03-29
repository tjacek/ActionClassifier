from keras.layers import Dense,Flatten
from keras import regularizers
import data.seqs,files,utils,spline,ens
import sim,ts_cnn,deep

class TS_SIM(object):
    def __init__(self, n_hidden=100,activ='relu',l1=0.01):
        self.activ=activ
        self.n_hidden=n_hidden
        self.l1=l1
        
    def __call__(self,model,params):
        n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
        deep.add_model_conv(model,n_kerns,kern_size,pool_size,self.activ)
        model.add(Flatten())
        reg=None if(self.l1 is None) else regularizers.l1(self.l1)
        model.add(Dense(self.n_hidden, activation=self.activ,
            name='hidden',kernel_regularizer=reg))
        return model

def binary_exp(in_path,dir_path,n_epochs=1000):
    files.make_dir(dir_path)
    input_paths=files.top_files("%s/seqs" % in_path)
    train=get_train()
    print(input_paths)
    ensemble1D(input_paths,dir_path,train,n_epochs)

def ensemble1D(input_paths,out_path,train,n_epochs=1000,size=64):
    extract= utils.Extract(data.seqs.read_seqs)
    ensemble=ens.ts_ensemble(train,extract,preproc=None)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def get_train():
	get_model=TS_SIM()
	get_cat=sim.all_cat
	read=data.seqs.read_seqs
	params={'input_shape':(64,100)}
	train_sim=sim.SimTrain(get_model,get_cat,read,2)
	def train(in_path,out_path,n_epochs):
		return train_sim(in_path,out_path,n_epochs,params=params)
	return train

in_path="../dtw_paper/MSR"
out_path="../dtw_paper/MHAD/sim_no_l1"
binary_exp(in_path,out_path,n_epochs=300)