import numpy as np
from keras.models import load_model
import data.imgs,data.feats,lstm

def agum_template(in_path,nn_path,out_path,seq_len=30,n_iters=10):
	dataset=data.imgs.read_frame_seqs(in_path,n_split=1)
	dataset.scale()
	train,test=dataset.split()
	model=load_model(nn_path)
	sample=lstm.MinLength(size=seq_len)
	agum_feats=data.feats.Feats()
	def prepare_seq(seq_i):
		seq_i=np.array(sample(seq_i))
		seq_i=np.expand_dims(seq_i,axis=0)
		return model.predict(seq_i)
	for name_i,seq_i in train.items():	
		for k in range(n_iters):
			name_k="%s_%d" % (name_i,k)
			agum_feats[name_k]=prepare_seq(seq_i)
	for name_i,seq_i in test.items():	
		agum_feats[name_i]=prepare_seq(seq_i)	
	agum_feats.save(out_path)

agum_template('../agum/frames','lstm4/nn','lstm4/agum_feats',seq_len=20)