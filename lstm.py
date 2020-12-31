import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import os.path
from keras.models import load_model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras import regularizers
import keras.utils,keras.optimizers
import data.imgs,utils,gen,ens

class MinLength(object):
	def __init__(self,size):
		self.size = size

	def __call__(self,frames):
		n_frames=len(frames)
		indexes=np.random.randint(n_frames,size=self.size)
		indexes=np.sort(indexes)
		return [frames[i] for i in indexes]

def train_lstm(in_path,out_path=None,n_epochs=200,seq_len=20):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	train.transform(MinLength(seq_len),single=False)
	train.scale()
	X,y=train.to_dataset()
	y=keras.utils.to_categorical(y)
	params={'n_cats':frames.n_cats(),"seq_len":seq_len,"drop":True}
	model=make_lstm(params)
#	raise Exception(params)
	model.fit(X,y,epochs=n_epochs,batch_size=8)
	if(out_path):
		model.save(out_path)

def train_gen_lstm(in_path,out_path=None,n_epochs=200,seq_len=20):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	train.scale()
	params={'n_cats':frames.n_cats(),"seq_len":seq_len,"drop":True}
	if(os.path.isfile(out_path)):
		model=load_model(out_path)
	else:
		model=make_lstm(params)
	seq_gen=gen.SeqGenerator(train,MinLength(params['seq_len']))
	model.fit_generator(seq_gen,epochs=n_epochs)
	if(out_path):
		model.save(out_path)

def make_lstm(params):
	input_shape=(params['seq_len'], 64, 64, 1)
	model=Sequential()
	model.add(TimeDistributed(Conv2D(32, (5, 5), padding='same'), input_shape=input_shape))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(32, (5, 5))))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4))))
	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(256)))
	if(params['drop']):
		model.add(TimeDistributed(Dropout(0.5)))	
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

def extract(in_path,nn_path,out_path,seq_len=20):
	read=data.imgs.read_frame_seqs
	def preproc(dataset):
		dataset.transform(MinLength(seq_len),single=False)
		dataset.scale()
	fun=utils.Extract(read,name="global_avg",preproc=preproc)
	fun(in_path,nn_path,out_path)

def binary_lstm(in_path,out_path,n_epochs=5,seq_len=20):
	n_cats=20
	binary_gen=dynamic_binary(in_path,n_epochs,seq_len)
	funcs=[[extract,["in_path","nn","feats"]]]
	dir_names=["feats"]
	arg_dict={'in_path':in_path}		
	binary_ens=ens.BinaryEns(binary_gen,funcs,dir_names)
	binary_ens(out_path,n_cats,arg_dict)

def static_binary(in_path,n_epochs=5,seq_len=20):
	dataset=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=dataset.split()
	train.transform(MinLength(seq_len),single=False)
	train.scale()
	params={'n_cats':2,"seq_len":seq_len,"drop":True}
	X,y=train.to_dataset()
	def binary_train(nn_path,i):
		y_i=ens.binarize(y,i)	
		model=make_lstm(params)
		model.fit(X,y_i,epochs=n_epochs,batch_size=8)
		model.save(nn_path)
	return binary_train		

def dynamic_binary(in_path,n_epochs=5,seq_len=20):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	train.scale()
	params={'n_cats':2,"seq_len":seq_len,"drop":True}
	seq_gen=gen.SeqGenerator(train,MinLength(params['seq_len']),binary=0)
	def binary_train(nn_path,i):
		seq_gen.binary=i
		model=make_lstm(params)
		model.fit_generator(seq_gen,epochs=n_epochs)
		model.save(nn_path)
	return binary_train

if __name__ == "__main__":
	train_gen_lstm('../agum/frames','lstm4/nn',n_epochs=200,seq_len=20)
	extract('../agum/frames','lstm4/nn','lstm4/feats',seq_len=20)
#	binary_lstm("../MSR/frames","../MSR/ens4",n_epochs=250,seq_len=20)