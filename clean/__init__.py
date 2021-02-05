import sys
sys.path.append("..")
import numpy as np
import cv2,pickle,os.path
from ast import literal_eval
from keras.models import load_model
import data.actions,data.imgs,gui

class TrainDataset(dict):
    def __init__(self, arg=[]):
        super(TrainDataset, self).__init__(arg)

    def init(self,data_dict):
        for name_i in data_dict.keys():
            self[name_i]=[30]

    def save(self,out_path):
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

def read(in_path):
    with open(in_path, 'rb') as handle:
        return pickle.load(handle)

class ActionState(object):
	def __init__(self, actions,dataset,cut,path="train_dataset"):
		self.actions=actions
		self.path=path
		self.dataset=dataset
		self.cut=cut

	def __getitem__(self,frame_i):
		value_i=self.dataset[frame_i]
		return str(value_i)

	def show(self,name_i,text_i):
		img_i=self.actions[name_i]
		position=literal_eval(text_i)
		self.dataset[name_i]=position
		img_i=self.cut(img_i,position)
		cv2.imshow(name_i,img_i)

	def keys(self):
		return list(self.actions.keys())

	def save(self,path_i):
		print("save at %s " % path_i)
		self.dataset.save(path_i)

def make_dataset_template(in_path,out_path,cut):
	action_imgs=data.actions.read_actions(in_path)
	if(out_path and os.path.isfile(out_path)):
		train_dataset=read(out_path)
	else:
		train_dataset=TrainDataset()
		train_dataset.init(action_imgs)
	state=ActionState(action_imgs,train_dataset,cut,out_path)
	gui.gui_exp(state)

def show_dataset_template(frame_path,dataset_path,out_path,cut):
	actions=data.actions.read_actions(frame_path)
	train_dataset=read(dataset_path)
	new_actions=data.actions.ActionImgs()
	for name_i,img_i in actions.items():
		position_i=train_dataset[name_i]
		new_actions[name_i]=cut(img_i,position_i)
	new_actions.save(out_path)

def get_dataset(frame_path,dataset_path):
	actions=data.actions.read_actions(frame_path)
	train_dataset=read(dataset_path)
	X,y=[],[]
	for name_i in actions.keys():
		X.append( actions[name_i])
		y.append(train_dataset[name_i][0])
	return np.array(X),y

def apply_model(frame_path,nn_path,out_path,cut):
	model=load_model(nn_path)
	frame_seqs=data.imgs.read_frame_seqs(frame_path,n_split=1)
	frame_seqs.scale((128,128))
	def helper(img_i):
		img_i=np.expand_dims(img_i,axis=0)
		position_i=model.predict(img_i)
		img_i=np.squeeze(img_i)
		img_i=cut(img_i,position_i)
		return img_i
	frame_seqs.transform(helper,new=False,single=True)
	frame_seqs.save(out_path)