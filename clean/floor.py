import sys
sys.path.append("..")
import numpy as np
import clean,cnn

def cut_floor(img_i,position):
	position=int(position[0])
	print(position)
	new_img=img_i.copy()
	position=img_i.shape[0]-position
	new_img[position:,:]=0
	return new_img

def make_dataset(in_path,out_path):
	clean.make_dataset_template(in_path,out_path,cut_floor)

def show_dataset(frame_path,dataset_path,out_path):
	clean.show_dataset_template(frame_path,dataset_path,out_path,cut_floor)

def train(frame_path,dataset_path,model_path,n_epochs=500):
	X,y=clean.get_dataset(frame_path,dataset_path)
	X=np.expand_dims(X,axis=-1)
	img_shape=X.shape[1:]#(*X.shape[1:],1)
	model=cnn.make_model(img_shape=img_shape,n_dense=1)
	model.fit(X,y,epochs=n_epochs)
	model.save(model_path)

def apply_model(frame_path,nn_path,out_path):
	clean.apply_model(frame_path,nn_path,out_path,cut_floor)

frame_path="../../clean/mean"
dataset_path="train_dataset"
model_path="cnn"
#make_dataset(frame_path,dataset_path)
#show_dataset(frame_path,dataset_path,"test")
#train(frame_path,dataset_path,model_path)
apply_model("../../clean/frames",model_path,"../../clean/clean_frames")