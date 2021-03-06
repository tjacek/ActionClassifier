import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.utils
from keras import regularizers
from keras.models import load_model
import numpy as np
import data.imgs,files,shutil

def train_model(in_path,nn_path="agum_nn",n_epochs=1000):
    frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
    frame_seqs.scale()
    train,test=frame_seqs.split()
    X,y,params=get_dataset(frame_seqs)
    model=make_conv(params)
    model.fit(X,y,epochs=n_epochs,batch_size=32)
    if(nn_path):
        model.save(nn_path)

def filtr_seqs(in_path,nn_path,out_path):
    frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
    scaled_seqs=frame_seqs.scale( new=True)
    model=load_model(nn_path)
    def helper(name_i):
        seq_i=scaled_seqs[name_i]
        result_i=model.predict(np.array(seq_i))
        no_action=np.argmax(result_i,axis=1)
        seq_i=[frame_j 
                for j,frame_j in enumerate(frame_seqs[name_i])
                    if(no_action[j]==1)]
        return seq_i
    frame_seqs={name_i:helper(name_i) 
            for name_i,seq_i in frame_seqs.items()}
    frame_seqs=data.imgs.FrameSeqs(frame_seqs)
    frame_seqs.save(out_path)

def get_dataset(frame_seqs):
    X,y=[],[]
    for name_i,seq_i in frame_seqs.items():
        middle=int(len(seq_i)/2)
        X+=[seq_i[0], seq_i[-1],seq_i[middle]]
        y+=[0,0,1]
    X,y=np.array(X),keras.utils.to_categorical(y)
    params={'dims': frame_seqs.dims(),'n_cats':y.shape[1]}
    return X,y,params

def make_conv(params):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=params['dims']))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu'))#,input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),))
    model.add(Dropout(0.5))
    model.add(Dense(units=params['n_cats'], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
              optimizer=keras.optimizers.Adadelta())
    model.summary()
    return model

def one_dict(paths,out_path=None):
    files.make_dir(out_path)
    for i,path_i in enumerate(paths):
        print(path_i)
        for sample_j in files.top_files(path_i):           
            name_j=sample_j.split('/')[-1]
            name_j,postfix=name_j.split('.')
            name_j=files.clean(name_j)
            if(i==0 or files.person_selector(name_j)):
                if(i==0):
                    out_i="%s/%s.%s" % (out_path,name_j,postfix)
                else:
                    out_i="%s/%s_%d.%s" % (out_path,name_j,i,postfix)
                shutil.copy(sample_j,out_i)

def agum_exp(in_path,n_epochs=100):
    paths=files.get_paths(in_path,['box','filtr_nn','frames'])
    train_model(paths["box"],paths["filtr_nn"],n_epochs)
    filtr_seqs(paths["box"],paths["filtr_nn"],paths["frames"])

if __name__ == "__main__":
    agum_exp("../3DHOI")