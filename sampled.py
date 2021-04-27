import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import sim,sim.dist,sim.imgs
import utils,data.imgs,files

def sampled_exp(in_path,out_path,n_samples=3,n_epochs=5):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["nn","seqs"])
	train(in_path,paths["nn"],n_samples,n_epochs)
	extract(in_path,paths['nn'],paths['seqs'])

def train(in_path,out_path=None,n_samples=3,n_epochs=5):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	def helper(seq_i):
		sample=sim.dist.get_sample(seq_i)
		return [seq_i[i] for i in sample(n_samples)]
	train.transform(helper,single=False)
	train.scale()
	X,y=pairs_dataset(train)
	params={ 'input_shape':train.dims(),"n_hidden":20}
	siamese_net,extractor=sim.build_siamese(params,sim.imgs.SimConv())
	siamese_net.fit(X,y,epochs=n_epochs,batch_size=64)
	if(out_path):
		extractor.save(out_path)

def pairs_dataset(train,k=5):
	X,y=[],[]
	names=sim.all_pairs(train.keys())
	names=sim.subsample(names,k)
	for name_i,name_j in names:
		cat_ij=name_i.get_cat()==name_j.get_cat()
		for x_i,x_j in zip(train[name_i],train[name_j]):
			X.append((x_i,x_j))
			y.append(cat_ij)
	X=np.array(X)
	X=[X[:,0],X[:,1]]
	return X,y

def extract(in_path,nn_path,out_path,n_channels=1):
	read=data.imgs.ReadFrames(n_channels)
	def preproc(dataset):
		dataset.scale()
	fun=utils.ExtractSeqs(read,"hidden",preproc)
	fun(in_path,nn_path,out_path)

sampled_exp("../MSR/full","test",n_epochs=500)