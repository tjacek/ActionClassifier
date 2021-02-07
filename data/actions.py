import numpy as np
import cv2
import files,data.imgs

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

	def add_dim(self):
		self.transform(lambda img_i: np.expand_dims(img_i,axis=-1))
	
	def transform(self,fun,copy=False):
		data_dict=ActionImgs() if(copy) else self
		for name_i,img_i in self.items():
			data_dict[name_i]=fun(img_i)
		return data_dict

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
		name_i=files.Name(path_i.split('/')[-1])
		name_i=name_i.clean()
		actions[name_i]=cv2.imread(path_i,cv2.IMREAD_GRAYSCALE)
	return actions

def get_actions(in_path,fun,out_path=None,dims=(64,64)):
	frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
	actions=ActionImgs()
	for name_i,seq_i in frame_seqs.items():
		actions[name_i]=fun(seq_i)
	if(dims):
		actions.scale(dims)
	if(out_path):
		actions.save(out_path)
	return actions

def tranform_actions(in_path,out_path,fun):
	actions=read_actions(in_path)
	actions.transform(fun)
	actions.save(out_path)