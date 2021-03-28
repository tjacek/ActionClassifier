import sys
sys.path.append("..")
import numpy as np
import ens,files,ts_cnn,deep
import data.seqs

def ensemble_agum(in_path,out_path,n_epochs=1000,size=128):
	model=ts_cnn.TS_CNN(dropout=0.5,optim_alg=deep.Nestrov())
	train,extract=ts_cnn.get_train(model)
	ensemble=ens.ts_ensemble(train,extract,preproc=norm_partial)	
	input_paths=files.top_files(in_path)
	arg_dict={"n_epochs":n_epochs,"size":size}
	files.make_dir(out_path)
	ensemble(input_paths,out_path,arg_dict)

def norm_full(in_path,out_path):
	seq_dict=data.seqs.read_seqs(in_path)
	X,y=seq_dict.to_dataset()
	X=X.reshape((X.shape[0]*X.shape[1],X.shape[2] ))
	mean,std=get_stats(X)
	def helper(seq_i):
		return (seq_i-mean)/std
	seq_dict.transform(helper)
	seq_dict.save(out_path)

def norm_partial(in_path,out_path):
	seq_dict=data.seqs.read_seqs(in_path)
	def helper(seq_i):
		mean,std=get_stats(seq_i)
		return (seq_i-mean)/std
	seq_dict.transform(helper)
	seq_dict.save(out_path)

def get_stats(X):
	mean=np.mean(X,axis=0)
	std=np.std(X,axis=0)
	std[std==0]=1.0
	return mean,std

if __name__ == "__main__":
    seq_path="../../dtw_paper/MSR/binary/seqs"
    ensemble_agum(seq_path,"agum8",n_epochs=1000,size=128)