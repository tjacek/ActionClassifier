import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras import regularizers
import keras.utils,keras.optimizers
import data.imgs,utils

class SeqGenerator(keras.utils.Sequence):
	def __init__(self,dataset,size=30,batch_size=16,n_agum=2):
		self.dataset=dataset
		self.n_cats=self.dataset.n_cats()
		self.agum=MinLength(size)
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

class MinLength(object):
	def __init__(self,size):
		self.size = size

	def __call__(self,frames):
		n_frames=len(frames)
		indexes=np.random.randint(n_frames,size=self.size)
		indexes=np.sort(indexes)
		return [frames[i] for i in indexes]

def train_lstm(in_path,out_path=None,n_epochs=200):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
#	train.transform(MinLength(30),single=False)#frames.min_len()))
	train.scale()
#	X,y=train.to_dataset()
#	y=keras.utils.to_categorical(y)
	params={'n_cats':frames.n_cats()}
	model=make_lstm(params)
#	model.fit(np.array(X),y,epochs=n_epochs)
	gen=SeqGenerator(train)
	model.fit_generator(gen,epochs=n_epochs)
	if(out_path):
		model.save(out_path)

def make_lstm(params):
	model=Sequential()
	model.add(TimeDistributed(Conv2D(32, (5, 5), padding='same'), input_shape=(30, 64, 64, 1)))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(32, (5, 5))))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4))))
	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(256)))
#	model.add(TimeDistributed(Dropout(0.25)))	
	model.add(TimeDistributed(Dense(128, name="first_dense",
	 kernel_regularizer=regularizers.l1(0.01))))

	model.add(LSTM(64, return_sequences=True, name="lstm_layer"));
#	model.add(TimeDistributed(Dense(n_cats), name="time_distr_dense_one"))
	model.add(GlobalAveragePooling1D(name="global_avg"))
	model.add(Dense(params['n_cats'],activation='softmax'))

	model.compile(loss='categorical_crossentropy',
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy'])
	model.summary()
	return model

def extract(in_path,nn_path,out_path):
	read=data.imgs.read_frame_seqs
	def preproc(dataset):
		dataset.transform(MinLength(30),single=False)
		dataset.scale()
	fun=utils.Extract(read,name="global_avg",preproc=preproc)
	fun(in_path,nn_path,out_path)

if __name__ == "__main__":
	train_lstm('../agum/frames','lstm/nn')
	extract('../agum/frames','lstm/nn','lstm/feats')