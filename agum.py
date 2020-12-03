from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.utils
from keras import regularizers
from keras.models import load_model
import numpy as np
import imgs

def train_model(in_path,nn_path="agum_nn",n_epochs=1000):
    frame_seqs=imgs.read_frame_seqs(in_path)
    train,test=frame_seqs.split()
    X,y,params=get_dataset(train)
    print(params)
    model=make_conv(params)
    model.fit(X,y,epochs=n_epochs,batch_size=32)
    if(nn_path):
        model.save(nn_path)

def filtr_seqs(in_path,nn_path,out_path):
    frame_seqs=imgs.read_frame_seqs(in_path)
    model=load_model(nn_path)
    def helper(seq_i):
        result_i=model.predict(np.array(seq_i))
        no_action=np.argmax(result_i,axis=1)
        seq_i=[frame_j 
                for j,frame_j in enumerate(seq_i)
                    if(no_action[j]==1)]
        return seq_i
    frame_seqs={name_i:helper(seq_i) 
            for name_i,seq_i in frame_seqs.items()}
    frame_seqs=imgs.FrameSeqs(frame_seqs)
    frame_seqs.save(out_path)

def get_dataset(frame_seqs):
    X,y=[],[]
    for name_i,seq_i in frame_seqs.items():
        middle=int(len(seq_i)/2)
        X+=[seq_i[0], seq_i[-1],seq_i[middle]]
        y+=[0,0,1]
    X,y=np.array(X),keras.utils.to_categorical(y)
    params={'dims':X.shape[1:],'n_cats':y.shape[1]}
    return X,y,params

def make_conv(params):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=params['dims']))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(16, kernel_size=(5, 5),
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

filtr_seqs("full","agum_nn","test2")