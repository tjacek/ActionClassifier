import numpy as np
import cv2
import imgs,files,feats
import sim.imgs

def diff_action(in_path,out_path):
	def helper(frames):
		size=len(frames)-1
		diff=[ np.abs(frames[i]-frames[i+1]) 
				for i in range(size)]
		return np.mean(diff,axis=0)
	get_actions(in_path,helper,out_path)

def mean_action(in_path,out_path):
	def helper(frames):
		return np.mean(frames,axis=0)
	get_actions(in_path,helper,out_path)

def action_one_shot(in_path,out_path=None,n_epochs=5):
    dtw_feats=read_actions(in_path)
    dtw_feats.transform(lambda img_i: np.expand_dims(img_i,axis=-1))
    def all_cat(name_i,name_j):
        return int(name_i.split('_')[0]==name_j.split('_')[0])
    make_model=sim.imgs.make_conv
    sim_train=sim.SimTrain(make_model,all_cat)
    params={'input_shape':(64,64,1)}
    sim_train(dtw_feats,out_path,n_epochs,params)

from keras.models import load_model
import binary,ens

def extract(in_path,nn_path,out_path):
    action_feats=read_actions(in_path)
    def helper(img_i):
    	img_i=np.expand_dims(img_i,axis=-1)
    	return np.expand_dims(img_i,axis=0)
    action_feats.transform(helper)
    extractor=load_model(nn_path)
    new_feats=feats.Feats()
    for name_i,img_i in action_feats.items():
    	new_feats[name_i]=extractor.predict(img_i)
    new_feats.save(out_path)

def binary_one_shot(in_path,out_path,n_epochs=5):
	n_cats=12
	dataset=read_actions(in_path)
	dataset.add_dim()
	get_cat=binary.BinaryCat()
	sim_nn=sim.SimTrain(sim.imgs.make_conv,get_cat)
	def binary_gen(nn_path,i):
		get_cat.cat=i
		sim_nn(dataset,nn_path,n_epochs)
	funcs=[[extract,["in_path","nn","feats"]]]
	dir_names=["feats"]
	arg_dict={'in_path':in_path}		
	binary_ens=ens.BinaryEns(binary_gen,funcs,dir_names)
	binary_ens(out_path,n_cats,arg_dict)

def action_img_exp(in_path,n_epochs=100):
	paths=files.get_paths(in_path,['frames','mean',"ens"])
#	mean_action(paths["frames"],paths["mean"])
	binary_one_shot(paths["mean"],paths["ens"],n_epochs)

if __name__ == "__main__":
	action_img_exp("../3DHOI_agum",100)