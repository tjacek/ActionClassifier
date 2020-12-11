import numpy as np
import cv2
import imgs,files

class ActionImgs(dict):
	def __init__(self, arg=[]):
		super(ActionImgs, self).__init__(arg)

	def save(self,out_path):
		files.make_dir(out_path)
		for name_i,img_i in self.items():
			out_i="%s/%s.png" % (out_path,name_i)
			cv2.imwrite(out_i, img_i)

def get_actions(in_path,fun,out_path=None):
	frame_seqs=imgs.read_frame_seqs(in_path,n_split=1)
	actions=ActionImgs()
	for name_i,seq_i in frame_seqs.items():
		actions[name_i]=fun(seq_i)
	if(out_path):
		actions.save(out_path)
	return actions

def mean_action(in_path,out_path):
	def helper(frames):
		return np.mean(frames,axis=0)
	get_actions(in_path,helper,out_path)

mean_action("../agum/box","../action/mean")