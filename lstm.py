import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras import regularizers
import keras.utils,keras.optimizers
import data.imgs,utils,gen

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
#	train.transform(MinLength(30),single=False)#frames.min_len()))
	train.scale()
#	X,y=train.to_dataset()
#	y=keras.utils.to_categorical(y)
	params={'n_cats':frames.n_cats(),"seq_len":seq_len}
	model=make_lstm(params)
#	model.fit(np.array(X),y,epochs=n_epochs)
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

def extract(in_path,nn_path,out_path,seq_len=20):
	read=data.imgs.read_frame_seqs
	def preproc(dataset):
		dataset.transform(MinLength(seq_len),single=False)
		dataset.scale()
	fun=utils.Extract(read,name="global_avg",preproc=preproc)
	fun(in_path,nn_path,out_path)

if __name__ == "__main__":
#	train_lstm('../MSR/frames','../MSR/nn',seq_len=20)
	extract('../MSR/frames','../MSR/nn','../MSR/feats',seq_len=20)