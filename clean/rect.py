import sys
sys.path.append("..")
import numpy as np 
import clean,files
import action_imgs,cnn,data.imgs

def rect_exp(action_path,frame_path,dataset_path,
				out_path,n_epochs=1500):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["nn","frames","actions"])
	train(action_path,dataset_path,paths["nn"],n_epochs)
	apply_model(frame_path,paths["nn"],paths["frames"])
	action_imgs.mean_action(paths["frames"],paths["actions"],None)

def cut_rect(img_i,position):
	if(np.product(position)==0):
		return img_i
	position=np.array(position)
	if(type(position)==np.ndarray):
		position=position.astype(int)
	position[position<0]=0
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

#def show_dataset(action_path,dataset_path,out_path):
#	clean.show_dataset_template(action_path,dataset_path,out_path,cut_rect)

#def apply_model(frame_path,nn_path,out_path):
#	clean.apply_model(frame_path,nn_path,out_path,cut_rect)

def dataset_cut(dataset_path,frame_path,out_path):
	dataset=clean.read(dataset_path)
	frame_seqs=data.imgs.read_frame_seqs(frame_path,n_split=1)
	new_seqs=data.imgs.FrameSeqs()
	for name_i in dataset.keys():
		seq_i=frame_seqs[name_i]
		position_i=dataset[name_i]
		new_seqs[name_i]=[cut_rect(frame_j,position_i) 
						for frame_j in seq_i]
	new_seqs.save(out_path)

if __name__ == "__main__":
	action_path="../../clean/exp2/actions"
	dataset_path="../../clean/exp2/dataset"
	make_dataset(action_path,dataset_path)