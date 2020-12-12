import numpy as np
import files,ens,feats,one_shot,sim

class BinaryCat(object):
	def __init__(self):
		self.cat=0

	def __call__(self,name_i,name_j):
		cat_i=int(name_i.split("_")[0])-1
		cat_j=int(name_j.split("_")[0])-1
		if(cat_i==self.cat or cat_j==self.cat):	
			print(cat_i)
			return int(cat_i==cat_j)
		else:
			return None

def binary_one_shot(in_path,ens_path,n_epochs=100):
    files.make_dir(ens_path)
    binary_sim(in_path,ens_path,n_epochs=n_epochs)
    funcs=[[one_shot.dtw_extract,["in_path","nn","feats"]]]
    dir_names=["feats"]
    ensemble=ens.EnsTransform(funcs,dir_names,"nn")
    arg_dict={'in_path':in_path}
    input_paths=[path_i for path_i in files.top_files("%s/nn"%ens_path)]
    ensemble(input_paths,ens_path, arg_dict)

def binary_sim(in_path,nn_path,n_epochs=5):
	dtw_feats=feats.read_feats(in_path)
	nn_path="%s/nn" % nn_path
	files.make_dir(nn_path)
	get_cat=BinaryCat()
	sim_nn=sim.SimTrain(one_shot.get_basic_model(),get_cat)
	for i in range(dtw_feats.n_cats()):
		nn_i="%s/%d" % (nn_path,i)
		sim_nn(dtw_feats,nn_i,n_epochs)
		get_cat.cat+=1

in_path=['../agum/max_z/dtw','../agum/corl/dtw','../agum/skew/dtw']
#["dtw/corl/feats","dtw/maxz/feats"]
binary_one_shot(in_path,"ens",n_epochs=300)