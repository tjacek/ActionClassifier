import sys
sys.path.append("..")
import numpy as np
import clean,data.actions,data.imgs

class CutBox(object):
	def __init__(self,binary):
		self.binary=binary

	def __call__(self,img_i):
		if(self.binary):
			img_i[img_i!=0]=200
		bound_i=frame_bounds(img_i)
		if(not bound_i is None):
			x0,x1=bound_i[2],bound_i[0]
			y0,y1=bound_i[3],bound_i[1]
			return img_i[x0:x1,y0:y1]
		return bound_i

def cut_actions(in_path,out_path,scale=None,binary=False):
	cut_box=CutBox(binary)
	data.actions.tranform_actions(in_path,out_path,cut_box,scale)

def cut_imgs(in_path,out_path):
	data.imgs.tranform_frames(in_path,out_path,cut_box)

def prepare_data(in_path,out_path):
	cut_box= CutBox(False)
	def helper(dataset):
		dataset.transform(cut_box,new=False,single=True)
		dataset.scale(dims=(64,64),new=False)
	data.imgs.tranform_frames(in_path,out_path,helper,whole=True)

def frame_bounds(frame_i):
    nonzero_i=np.array(np.nonzero(frame_i))
    if(nonzero_i.shape[1]==0):
        return None
    f_max=np.max(nonzero_i,axis=1)
    f_min=np.min(nonzero_i,axis=1)
    return np.concatenate([f_max,f_min])

def clean_names(in_path,out_path):
	dataset=clean.read(in_path)
	dataset={ name_i.clean():data_i 
		for name_i,data_i in dataset.items()}
	dataset=clean.TrainDataset(dataset)
	dataset.save(out_path)

in_path="../../clean/exp3/frames"
out_path="../../clean/exp3/final"
prepare_data(in_path,out_path)