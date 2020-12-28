import numpy as np
import keras.utils
import utils

class SeqGenerator(keras.utils.Sequence):
	def __init__(self,dataset,agum,batch_size=16,n_agum=2):
		self.dataset=dataset
		self.n_cats=self.dataset.n_cats()
#		self.agum=MinLength(size)
		self.agum=agum
		self.batch_size=batch_size
		self.names=self.dataset.names()
		self.n_agum=n_agum

	def __len__(self):
		return int(len(self.dataset)/self.batch_size)+1

	def __getitem__(self, i):
		names_i=self.names[i*self.batch_size:(i+1)*self.batch_size]
		X,y=[],[]
		for name_j in names_i:
			for k in range(self.n_agum):
				X.append(self.agum(self.dataset[name_j]))
				y.append(name_j.get_cat())
		y=utils.to_one_hot(y,self.n_cats)
		return np.array(X),y

	def on_epoch_end(self):
		np.random.shuffle(self.names)