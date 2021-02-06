import sys
sys.path.append("..")
import numpy as np
import data.actions

def cut_box(in_path,out_path):
	def fun(img_i):
		img_i[img_i!=0]=200
		bound_i=frame_bounds(img_i)
		if(not bound_i is None):
			x0,x1=bound_i[2],bound_i[0]
			y0,y1=bound_i[3],bound_i[1]
			return img_i[x0:x1,y0:y1]
		return bound_i
	data.actions.tranform_actions(in_path,out_path,fun)

def frame_bounds(frame_i):
    nonzero_i=np.array(np.nonzero(frame_i))
    if(nonzero_i.shape[1]==0):
        return None
    f_max=np.max(nonzero_i,axis=1)
    f_min=np.min(nonzero_i,axis=1)
    return np.concatenate([f_max,f_min])

in_path="../../clean/rect/actions"
out_path="../../clean/rect/bound"
cut_box(in_path,out_path)