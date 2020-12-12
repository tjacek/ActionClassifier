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

def binary_one_shot(in_path,out_path,n_epochs=5):
	n_cats=20
	dataset=feats.read_feats(in_path)
	get_cat=BinaryCat()
	sim_nn=sim.SimTrain(one_shot.get_basic_model(),get_cat)
	def binary_gen(nn_path,i):
		get_cat.cat=i
		sim_nn(dataset,nn_path,n_epochs)
	funcs=[[one_shot.dtw_extract,["in_path","nn","feats"]]]
	dir_names=["feats"]
	arg_dict={'in_path':in_path}		
	binary_ens=ens.BinaryEns(binary_gen,funcs,dir_names)
	binary_ens(out_path,n_cats,arg_dict)

in_path=['../agum/max_z/dtw','../agum/corl/dtw','../agum/skew/dtw']
#["dtw/corl/feats","dtw/maxz/feats"]
binary_one_shot(in_path,"ens",n_epochs=100)