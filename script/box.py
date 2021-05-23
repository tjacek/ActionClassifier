import sys
sys.path.append("..")
import numpy as np 
import data.imgs

def get_box(in_path,out_path):
	frames=data.imgs.read_frame_seqs(in_path)
	frames.transform(extract_box,new=False,single=True)
	frames.save(out_path)

def extract_box(frame_i):
	f_max,f_min=frame_bounds(frame_i)
	box_i=frame_i[f_min[0]:f_max[0],f_min[1]:f_max[1]]	
	print(frame_i.shape)
	return box_i

def frame_bounds(frame_i):
    nonzero_i=np.array(np.nonzero(frame_i))
#    if(nonzero_i.shape[1]==0):
#        return None
    f_max=np.max(nonzero_i,axis=1)
    f_min=np.min(nonzero_i,axis=1)
    return f_max,f_min

depth_path="../../forth/Depth_227x227x3"
box_path="../../forth/box"
get_box(depth_path,box_path)