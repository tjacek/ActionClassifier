import sys
sys.path.append("..")
import numpy as np
import cv2
import clean,data.actions,data.imgs

def cut_actions(in_path,out_path,scale=None):
	data.actions.tranform_actions(in_path,out_path,cut_box,scale)

def cut_imgs(in_path,out_path):
	data.imgs.tranform_frames(in_path,out_path,cut_box)

def prepare_data(in_path,out_path):
	def helper(img_i):
		img_i=cut_box(img_i)
		if(img_i.shape[0]==0):
			return None
		print(img_i.shape)
		img_i=cv2.resize(img_i,dsize=(64,64),interpolation=cv2.INTER_CUBIC)
		print(img_i.shape)
		return img_i
	data.imgs.tranform_frames(in_path,out_path,helper)

def cut_box(img_i):
#	img_i[img_i!=0]=200
	bound_i=frame_bounds(img_i)
	if(not bound_i is None):
		x0,x1=bound_i[2],bound_i[0]
		y0,y1=bound_i[3],bound_i[1]
		return img_i[x0:x1,y0:y1]
	return bound_i

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

prepare_data("../../clean2/frames","../../clean2/base")
#cut_actions("../../clean3/actions","../../clean3/final",(64,64))
#clean_names("../../clean/rect/dataset","../../clean/rect/new_dataset")