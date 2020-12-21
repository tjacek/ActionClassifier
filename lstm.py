import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
import data.imgs

def train_lstm(in_path):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	frames.scale()
	print(frames.dims())
	make_lstm()

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

	model.summary()
	return model

if __name__ == "__main__":
	train_lstm('../3DHOI/frames')