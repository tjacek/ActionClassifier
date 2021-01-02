import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras import regularizers
import data.imgs,utils,files

def ae_exp(in_path,out_path,n_epochs=5):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["ae","feats"])
	train_ae(in_path,paths['ae'],n_epochs)
	extract(in_path,paths['ae'],paths['feats'])

def train_ae(in_path,out_path,n_epochs=5):
	frames=data.imgs.read_frame_seqs(in_path,n_split=1)
	train,test=frames.split()
	train.scale()
	X=to_dataset(train)
	params={'n_channels':X.shape[-1],"dim":(X.shape[1],X.shape[2]),'scale':(2,2)}
	autoencoder,recon=make_autoencoder(params)
	autoencoder.summary()
	autoencoder.fit(X,X,epochs=n_epochs,batch_size=16)
	autoencoder.save(out_path)

def to_dataset(data_dict):
	X=[]
	for name_i,seq_i in data_dict.items():
		X+=seq_i
	return np.array(X)

def extract(in_path,nn_path,out_path):
	def preproc(dataset):
		dataset.scale()
	fun=utils.ExtractSeqs(name="hidden",preproc=preproc)
	fun(in_path,nn_path,out_path)

def make_autoencoder(params):
    scale = params["scale"] if("scale" in params) else (2,2)
    x,y=params["dim"]
    input_img = Input(shape=(x,y, params['n_channels']))
    n_kerns=32
    x = Conv2D(n_kerns, (5, 5), activation='relu',padding='same')(input_img)
    x = MaxPooling2D(scale)(x)
    x = Conv2D(16, (5, 5), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    shape = K.int_shape(x)
    x=Flatten()(x)
    encoded=Dense(100,name='hidden',kernel_regularizer=regularizers.l1(0.01))(x)    
    x = Dense(shape[1]*shape[2]*shape[3])(encoded)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (5, 5), activation='relu',padding='same')(x)
    x = UpSampling2D(scale)(x)
    x = Conv2DTranspose(n_kerns, (5, 5), activation='relu',padding='same')(x)
    
    x=Conv2DTranspose(filters=params['n_channels'],kernel_size=n_kerns,padding='same')(x)
    recon=Model(input_img,encoded)
    autoencoder = Model(input_img, x)

    autoencoder.compile(optimizer='adam',
                      loss='mean_squared_error')#CustomLoss(autoencoder)
    return autoencoder,recon

ae_exp("../3DHOI/box","../3DHOI/ae",n_epochs=5)
#train_ae("../3DHOI/box","test")
#extract("../3DHOI/box","test","ae_feats")