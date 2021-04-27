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
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import keras.utils,keras.optimizers
import data.imgs,utils,gen,ens,files,deep

class MinLength(object):
	def __init__(self,size):
		self.size = size

	def __call__(self,frames):
		n_frames=len(frames)
		indexes=np.random.randint(n_frames,size=self.size)
		indexes=np.sort(indexes)
		return [frames[i] for i in indexes]

class FRAME_LSTM(object):
	def __init__(self,dropout=0.5,activ='relu',batch=True,optim_alg=None,l1=0.01):
		if(optim_alg is None):
			optim_alg=deep.Adam(0.00001)
		self.dropout=dropout
		self.activ=activ
		self.batch=batch
		self.l1=l1
		self.optim_alg=optim_alg

	def __call__(self,params):
		input_shape= (params['seq_len'],*params['dims']) 
		model=Sequential()
		n_kern,kern_size,pool_size=[64,64,64],[(5,5),(5,5),(5,5)],[(2,2),(2,2),(2,2)]
		deep. lstm_cnn(model,n_kern,kern_size,pool_size,self.activ,input_shape)
		model.add(TimeDistributed(Flatten()))
		model.add(TimeDistributed(Dense(256)))
		if( not (self.dropout is None)):
			model.add(TimeDistributed(Dropout(self.dropout)))	
		reg=None if(self.l1  is None) else regularizers.l1(self.l1)
		model.add(TimeDistributed(Dense(128, name="first_dense",
				kernel_regularizer=reg)))

		model.add(LSTM(64, return_sequences=True, name="lstm_layer"));
		
		if(self.batch):
			model.add(GlobalAveragePooling1D(name="prebatch"))
			model.add(BatchNormalization(name="global_avg"))
		else:
			model.add(GlobalAveragePooling1D(name="global_avg"))
		model.add(Dense(params['n_cats'],activation='softmax'))

		model.compile(loss='categorical_crossentropy',
			optimizer=self.optim_alg(),#keras.optimizers.Adadelta(),
			metrics=['accuracy'])
		model.summary()
		return model

def train_lstm(in_path,out_path=None,n_epochs=200,seq_len=20,n_channels=3,static=True):
	frames=data.imgs.read_frame_seqs(in_path,n_split=n_channels)
	train,test=frames.split()
	if(static):
		train.transform(MinLength(seq_len),single=False)
	else:
		seq_gen=gen.SeqGenerator(train,MinLength(seq_len))
	train.scale()
	params={'n_cats':frames.n_cats(),"seq_len":seq_len,"dims":train.dims()}
	make_lstm=FRAME_LSTM()
	model=make_lstm(params)
	if(static):
		X,y=train.to_dataset()
		y=keras.utils.to_categorical(y)
		model.fit(X,y,epochs=n_epochs,batch_size=8)
	else:
		model.fit_generator(seq_gen,epochs=n_epochs)
	if(out_path):
		model.save(out_path)

def extract(in_path,nn_path,out_path,seq_len=30):
	read=data.imgs.ReadFrames(3)
	def preproc(dataset):
		dataset.transform(MinLength(seq_len),single=False)
		dataset.scale()
	fun=utils.Extract(read,name="global_avg",preproc=preproc)
	fun(in_path,nn_path,out_path)

def binary_lstm(in_path,out_path,n_epochs=5,seq_len=20):
	n_cats=20
	binary_gen=get_binary(in_path,n_epochs,seq_len)
	funcs=[[extract,["in_path","nn","feats"]]]
	dir_names=["feats"]
	arg_dict={'in_path':in_path}		
	binary_ens=ens.BinaryEns(binary_gen,funcs,dir_names)
	binary_ens(out_path,n_cats,arg_dict)

def get_binary(in_path,n_epochs=5,seq_len=20,n_channels=3,static=True):
	dataset=data.imgs.read_frame_seqs(in_path,n_split=n_channels)
	train,test=dataset.split()
	if(static):
		train.transform(MinLength(seq_len),single=False)
	else:
		seq_gen=gen.SeqGenerator(train,MinLength(seq_len),batch_size=4,n_agum=1,binary=0)
	train.scale()
	params={'n_cats':2,"seq_len":seq_len,"dims":train.dims(),"drop":False}
	X,y=train.to_dataset()
	make_lstm=FRAME_LSTM()
	def binary_train(nn_path,i):
		y_i=ens.binarize(y,i)	
		model=make_lstm(params)
		if(static):
			model.fit(X,y_i,epochs=n_epochs,batch_size=4)
		else:
			model.fit_generator(seq_gen,epochs=n_epochs)
		model.save(nn_path)
	return binary_train		

def lstm_exp(in_path,out_path,n_epochs=200,seq_len=20,static=True):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["nn","feats"])
	train_lstm(in_path,paths['nn'],n_epochs,seq_len,static=static)
	extract(in_path,paths['nn'],paths['feats'],seq_len)

if __name__ == "__main__":
	binary_lstm('../MSR/full','../MSR/lstm5',n_epochs=200,seq_len=30)
#	lstm_exp('../MSR/full','../MSR/lstm_all3',n_epochs=200,seq_len=30)
#	extract('../3DHOI/frames','../3DHOI/nn','../3DHOI/feats',seq_len=20)
#	binary_lstm("../MSR/frames","../MSR/ens4",n_epochs=250,seq_len=20)