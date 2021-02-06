import sys
sys.path.append("..")
import numpy as np 
import clean,files

def cut_rect(img_i,position):
	if(np.product(position)==0):
		return img_i
	x0,y0=position[0],position[1]
	x1,y1=x0+position[2],y0+position[3]
	img_i=img_i.copy()
	img_i[x0:x1,y0:y1]=0
	return img_i

def make_dataset(in_path,out_path):
	dir_path="/".join(out_path.split('/')[:-1])
	files.make_dir(dir_path)
	value=[0,0,0,0]
	clean.make_dataset_template(in_path,out_path,cut_rect,value)

def show_dataset(action_path,dataset_path,out_path):
	clean.show_dataset_template(action_path,dataset_path,out_path,cut_rect)


action_path="../../clean/floor/actions"
dataset_path="../../clean/rect/dataset"
out_path="../../clean/rect/actions"
#make_dataset(action_path,dataset_path)
show_dataset(action_path,dataset_path,out_path)