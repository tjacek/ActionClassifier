import sys
sys.path.append("..")
import numpy as np 
import clean,files
import action_imgs,cnn

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
	position=position.astype(int)
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

def train(frame_path,dataset_path,model_path,n_epochs=500):
	X,y=clean.get_dataset(frame_path,dataset_path)
	X=np.expand_dims(X,axis=-1)
	img_shape=X.shape[1:]
	model=cnn.make_model(img_shape=img_shape,n_dense=4)
	model.fit(X,y,epochs=n_epochs,batch_size=16)
	model.save(model_path)

def apply_model(frame_path,nn_path,out_path):
	clean.apply_model(frame_path,nn_path,out_path,cut_rect)

action_path="../../clean/floor/actions"
dataset_path="../../clean/rect/dataset"
frame_path="../../clean/floor/frames"
action_path="../../clean/rect/actions"
out_path="../../clean/rect/frames"
#make_dataset(action_path,dataset_path)
#show_dataset(action_path,dataset_path,action_path)
rect_exp(action_path,frame_path,dataset_path,out_path)