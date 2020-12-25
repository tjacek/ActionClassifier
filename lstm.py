import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
import keras.utils,keras.optimizers
import data.imgs

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
	train.transform(MinLength(30),single=False)#frames.min_len()))
	train.save("lstm/test")
	train.scale()
	X,y=train.to_dataset()
	y=keras.utils.to_categorical(y)
	model=make_lstm()
	model.fit(np.array(X),y,epochs=n_epochs)


def make_lstm():
	n_cats=20
	model=Sequential()
	model.add(TimeDistributed(Conv2D(32, (5, 5), padding='same'), input_shape=(30, 64, 64, 1)))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Conv2D(32, (5, 5))))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4))))
	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(256)))
#	model.add(TimeDistributed(Dropout(0.25)))	
	model.add(TimeDistributed(Dense(128, name="first_dense" )))

	model.add(LSTM(64, return_sequences=True, name="lstm_layer"));
#	model.add(Dropout(0.5))
#	model.add(Dense(64, activation='relu'))
#	model.add(Dense(n_cats, activation='softmax'))
	model.add(TimeDistributed(Dense(n_cats), name="time_distr_dense_one"))
	model.add(GlobalAveragePooling1D(name="global_avg"))
	model.add(Dense(n_cats,activation='softmax'))

	model.compile(loss='categorical_crossentropy',
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy'])
	model.summary()
	return model

if __name__ == "__main__":
	train_lstm('../agum/frames','lstm/nn')