import numpy as np
import data.imgs,sim

def train(in_path,n_samples=3):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	def helper(seq_i):
		sample=get_sample(seq_i)
		return [seq_i[i] for i in sample(n_samples)]
	train.transform(helper,single=False)
	print(train.n_frames())

def to_dataset(data_dict,n_sampled):
	X,y=[],[]
	for name_i,name_j in all_pairs(data_dict.keys()):
		y.append(int(name_i.get_cat()==name_j.get_cat()))
	return X,y

def get_sample(seq_i):
	if(type(seq_i)==list):
		size=len(seq_i)
	else:	
		size=seq_i.shape[0]
	dist_i=get_dist(size)
	def sample_helper(n):   
		return np.random.choice(np.arange(size),n,p=dist_i)
	return sample_helper

def get_dist(n):
    inc,dec=np.arange(n),np.flip(np.arange(n))
    dist=np.amin(np.array([inc,dec]),axis=0)
    dist=dist.astype(float)
    dist=dist**2
    if(np.sum(dist)==0):
        dist.fill(1.0)
    dist/=np.sum(dist)
    return dist


train("../ICSS_exp/MSR/frames")