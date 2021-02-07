import sys
sys.path.append("..")
import numpy as np
import clean,cnn
import action_imgs,files

def floor_exp(action_path,frame_path,dataset_path,
				out_path,n_epochs=750):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["nn","frames","actions"])
	train(action_path,dataset_path,paths["nn"],n_epochs)
	apply_model(frame_path,paths["nn"],paths["frames"])
	action_imgs.mean_action(paths["frames"],paths["actions"],None)

def cut_floor(img_i,position):
	position=int(position[0])
	print(position)
	new_img=img_i.copy()
	position=img_i.shape[0]-position
	new_img[position:,:]=0
	return new_img

def make_dataset(in_path,out_path):
	dir_path="/".join(out_path.split('/')[:-1])
	files.make_dir(dir_path)
	clean.make_dataset_template(in_path,out_path,cut_floor,[30])

def show_dataset(frame_path,dataset_path,out_path):
	clean.show_dataset_template(frame_path,dataset_path,out_path,cut_floor)

def train(frame_path,dataset_path,model_path,n_epochs=500):
	X,y=clean.get_dataset(frame_path,dataset_path)
	X=np.expand_dims(X,axis=-1)
	img_shape=X.shape[1:]
	model=cnn.make_model(img_shape=img_shape,n_dense=1)
	model.fit(X,y,epochs=n_epochs,batch_size=16)
	model.save(model_path)

def apply_model(frame_path,nn_path,out_path):
	clean.apply_model(frame_path,nn_path,out_path,cut_floor)

action_path="../../clean/mean"
frame_path="../../clean/scaled"
dataset_path="../../clean/floor/dataset"
out_path="../../clean/floor"

#make_dataset(action_path,dataset_path)
#show_dataset(frame_path,dataset_path,"test")
floor_exp(action_path,frame_path,dataset_path,out_path)