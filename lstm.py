import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
import keras.utils
import data.imgs

class MinLength(object):
	def __init__(self,size):
		self.size = size

	def __call__(self,frames):
		n_frames=len(frames)
		indexes=np.random.randint(n_frames,size=self.size)
		indexes=np.sort(indexes)
		return [frames[i] for i in indexes]

def train_lstm(in_path):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	train.transform(MinLength(10),single=False)#frames.min_len()))
	train.scale()
	X,y=train.to_dataset()
	y=keras.utils.to_categorical(y)
	model=make_lstm()
#	print(model.get_input_shape_at(0))
	model.fit(np.array(X),y,epochs=5)

def make_lstm():
	n_cats=20
	model=Sequential()
	model.add(TimeDistributed(Conv2D(32, (5, 5), padding='same'), input_shape=(10, 64, 64, 1)))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(32, (5, 5))))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4))))
	model.add(TimeDistributed(Dropout(0.25)))
	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(512)))
	
	model.add(TimeDistributed(Dense(35, name="first_dense" )))

	model.add(LSTM(20, return_sequences=True, name="lstm_layer"));

	model.add(TimeDistributed(Dense(n_cats), name="time_distr_dense_one"))
	model.add(GlobalAveragePooling1D(name="global_avg"))
    
	model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop')
	model.summary()
	return model

if __name__ == "__main__":
	train_lstm('../agum/box')