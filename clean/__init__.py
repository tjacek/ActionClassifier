import sys
sys.path.append("..")
import cv2,pickle
from ast import literal_eval 
import data.actions,gui

class TrainDataset(dict):
    def __init__(self, arg=[]):
        super(TrainDataset, self).__init__(arg)

    def init(self,data_dict):
        for name_i in data_dict.keys():
            self[name_i]=[0]

    def save(self,out_path):
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

class ActionState(object):
	def __init__(self, actions,dataset,path="train_dataset"):
		self.actions=actions
		self.path=path
		self.dataset=dataset

	def __getitem__(self,frame_i):
		value_i=self.dataset[frame_i]
		return str(value_i)

	def show(self,name_i,text_i):
		img_i=self.actions[name_i]
		position=literal_eval(text_i)
		self.dataset[name_i]=position
		img_i=cut(img_i,position)
		cv2.imshow(name_i,img_i)

	def keys(self):
		return list(self.actions.keys())

	def save(self,path_i):
		print("save at %s " % path_i)
		self.dataset.save(path_i)

def cut(img_i,position):
	position=int(position[0])
	new_img=img_i.copy()
	position=img_i.shape[0]-position
	new_img[position:,:]=0
	return new_img

def make_dataset(in_path):
	action_imgs=data.actions.read_actions(in_path)
	train_dataset=TrainDataset()
	train_dataset.init(action_imgs)
	state=ActionState(action_imgs,train_dataset)
	gui.gui_exp(state)
#	print(len(action_imgs))

in_path="../../clean/mean"
make_dataset(in_path)
#data.imgs.rescale_seqs("../../clean/frames","../../clean/scaled",dims=(128,128))