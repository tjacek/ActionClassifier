import numpy as np
import keras.utils,math
import utils

class SeqGenerator(keras.utils.Sequence):
	def __init__(self,dataset,agum,batch_size=8,n_agum=1,binary=None):
		self.dataset=dataset
#		self.n_cats=self.dataset.n_cats()
		self.agum=agum
		self.batch_size=batch_size
		self.names=self.dataset.names()
		self.n_agum=n_agum
		self.binary=binary

	def __len__(self):
		return math.ceil(len(self.dataset)/self.batch_size)

	def __getitem__(self, i):
		names_i=self.names[i*self.batch_size:(i+1)*self.batch_size]
		X,y=[],[]
		for name_j in names_i:
			for k in range(self.n_agum):
				X.append(self.agum(self.dataset[name_j]))
				y.append(self.get_cat(name_j))
		X,y=np.array(X),utils.to_one_hot(y,self.n_cats())
		return X,y

	def on_epoch_end(self):
		np.random.shuffle(self.names)

	def n_cats(self):
		if( self.binary is None):
			return self.dataset.n_cats()
		return 2

	def get_cat(self,name_j):
		if( self.binary is None):
			return name_j.get_cat()
		return name_j.get_cat()==self.binary