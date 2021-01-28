import numpy as np
import data.imgs,sim,sim.dist,sim.imgs

def train(in_path,n_samples=3):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	def helper(seq_i):
		sample=sim.dist.get_sample(seq_i)
		return [seq_i[i] for i in sample(n_samples)]
	train.transform(helper,single=False)
	X,y=pairs_dataset(train)
	params={ 'input_shape':(*train.dims(),1),"n_hidden":20}
	model=sim.build_siamese(params,sim.imgs.make_conv)

def pairs_dataset(train):
	X,y=[],[]
	for name_i,name_j in sim.all_pairs(train.keys()):
		cat_ij=name_i.get_cat()==name_j.get_cat()
		for x_i,x_j in zip(train[name_i],train[name_j]):
			X.append((x_i,x_j))
			y.append(cat_ij)
	return X,y

train("../ICSS_exp/MSR/frames")