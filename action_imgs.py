import numpy as np
import cv2
import imgs,files,feats
import sim.imgs

class ActionImgs(dict):
	def __init__(self, arg=[]):
		super(ActionImgs, self).__init__(arg)

	def dim(self):
		return list(self.values())[0].shape
    
	def scale(self,dims=(64,64)):
		def helper(img_i):
			return cv2.resize(img_i,dsize=dims,
							interpolation=cv2.INTER_CUBIC)
		self.transform(helper)

	def transform(self,fun):
		for name_i,img_i in self.items():
			self[name_i]=fun(img_i)

	def names(self):
		return sorted(self.keys(),key=files.natural_keys) 

	def split(self,selector=None):
		train,test=files.split(self,selector)
		return ActionImgs(train),ActionImgs(test)

	def save(self,out_path):
		files.make_dir(out_path)
		for name_i,img_i in self.items():
			out_i="%s/%s.png" % (out_path,name_i)
			cv2.imwrite(out_i, img_i)

def read_actions(in_path):
	actions=ActionImgs()
	for path_i in files.top_files(in_path):
		name_i=files.clean(path_i.split('/')[-1])
		actions[name_i]=cv2.imread(path_i,cv2.IMREAD_GRAYSCALE)
	return actions

def get_actions(in_path,fun,out_path=None):
	frame_seqs=imgs.read_frame_seqs(in_path,n_split=1)
	actions=ActionImgs()
	for name_i,seq_i in frame_seqs.items():
		actions[name_i]=fun(seq_i)
	actions.scale()
	if(out_path):
		actions.save(out_path)
	return actions

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

#mean_action("../agum/box","../action/mean")
#action_one_shot("../action/mean","../action/nn")
extract("../action/mean","../action/nn","../action/feats")