import sys
sys.path.append("..")
import clean

def cut_floor(img_i,position):
	position=int(position[0])
	new_img=img_i.copy()
	position=img_i.shape[0]-position
	new_img[position:,:]=0
	return new_img

def make_dataset(in_path,out_path):
	clean.make_dataset_template(in_path,out_path,cut_floor)

def show_dataset(frame_path,dataset_path,out_path):
	clean.show_dataset_template(frame_path,dataset_path,out_path,cut_floor)

in_path="../../clean/mean"
out_path="train_dataset"
#make_dataset(in_path,out_path)
show_dataset(in_path,out_path,"test")