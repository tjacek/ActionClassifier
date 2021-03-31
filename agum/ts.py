import sys
sys.path.append("..")
import numpy as np
import data.seqs,files,spline
import deep,ts_cnn,ens

class Agum(object):
	def __init__(self, agum_fun):
		self.agum_fun=agum_fun
		
	def __call__(self,in_path,out_path):
		seq_dict=data.seqs.read_seqs(in_path)
		train,test=seq_dict.split()
		agum_seqs=data.seqs.Seqs()
		for name_i,seq_i in train.items():
			samples_i=self.get_samples(seq_i)
			for k,sample_k in enumerate(samples_i):
				name_k="%s_%d" % (name_i,k)
				agum_seqs[name_k]=sample_k
			agum_seqs[name_i]=seq_i
		full_seqs=data.seqs.Seqs({**agum_seqs, **test})
		full_seqs.save(out_path)

	def get_samples(self,seq_i):
		samples=[]
		for fun_k in self.agum_fun:
			samples+=fun_k(seq_i)
		return samples

def ts_scale(seq_i):
	return [2*seq_i,0.5*seq_i]

def ts_reverse(seq_i):
	return [np.flip(seq_i,axis=0)]

def ts_binary(seq_i):
#	seq_i=seq_i.astype(int)
#	seq_i=seq_i.astype(float)
	seq_i=np.sign(seq_i)
	return [seq_i]

def agum_exp(input_path,out_path,n_epochs=1000):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["spline","agum","nn","feats"])
	spline.upsample(input_path,paths["spline"],size=64)
	agum=Agum([ts_scale,magnitude_warp])
	agum(paths["spline"],paths["agum"])
	train,extract=ts_cnn.get_train()
	train(paths["agum"],paths["nn"],n_epochs)
	extract(paths["agum"],paths["nn"],paths["feats"])

def ensemble_agum(in_path,out_path,n_epochs=1000,size=128):
	model=ts_cnn.TS_CNN(dropout=0.5,l1=None,optim_alg=deep.Nestrov())
	train,extract=ts_cnn.get_train(model)
	agum=Agum([ts_binary])
	funcs=[ [spline.upsample,["seqs","spline","size"]],
            [agum,["spline","agum"]],
            [train,["agum","nn","n_epochs"]],
            [extract,["agum","nn","feats"]]]
	dir_names=["spline","agum","nn","feats"]
	ensemble=ens.EnsTransform(funcs,dir_names)
	arg_dict={"n_epochs":n_epochs,"size":size}
	input_paths=files.top_files(in_path)
	files.make_dir(out_path)
	ensemble(input_paths,out_path,arg_dict)    

seq_path="../../dtw_paper/MSR/binary/seqs"
#ensemble_exp(seq_path,"ens_test")
ensemble_agum(seq_path,"agum10",n_epochs=1000)