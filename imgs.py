#import keras,keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.models import load_model
from keras import regularizers
import numpy as np
import cv2
import keras.utils
import files

class FrameSeqs(dict):
    def __init__(self, args=[]):
        super(FrameSeqs, self).__init__(args)

    def dim(self):
        return list(self.values())[0][0].shape

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return FrameSeqs(train),FrameSeqs(test)

def read_frame_seqs(in_path):
    frame_seqs=FrameSeqs()
    for path_i in files.top_files(in_path):
        name_i=files.clean(path_i.split('/')[-1])
        frames=[ read_frame(path_j,n_split=3) 
                for path_j in files.top_files(path_i)]
        frame_seqs[name_i]=frames
    return frame_seqs

def read_frame(in_path,n_split=3):
    frame_ij=cv2.imread(in_path,cv2.IMREAD_GRAYSCALE)
    return np.array(np.vsplit(frame_ij,n_split)).T

def train_model(in_path,nn_path=None,n_epochs=5):
    frame_seqs=read_frame_seqs(in_path)
    train,test=frame_seqs.split()
    X,y,params=get_dataset(train)
    print(params)
    model=make_conv(params)
    model.fit(X,y,epochs=n_epochs,batch_size=32)
    if(nn_path):
        model.save(nn_path)

def get_dataset(frame_seqs):
    X,y=[],[]
    for name_i,seq_i in frame_seqs.items():
        middle=int(len(seq_i)/2)
        X+=[seq_i[0], seq_i[-1],seq_i[middle]]
        y+=[0,0,1]
    X,y=np.array(X),keras.utils.to_categorical(y)
#    params={'ts_len':X.shape[1],'n_feats':X.shape[2],
#                'n_cats':y.shape[1],'n_channals':X.shape[-1]}
    params={'dims':X.shape[1:],'n_cats':y.shape[1]}
    return X,y,params
#def extract_frame_feats(in_path,nn_path,out_path):
#    model=load_model(nn_path)
#    action_dirs=data.top_files(in_path)
#    action_dirs.sort(key=data.natural_keys)
#    X,y=[],[]
#    data.make_dir(out_path)
#    for action_i in action_dirs:
#        frame_path=data.top_files(action_i)
#        frame_path.sort(key=data.natural_keys)
#        def frame_helper(path_ij):
#            frame_ij=read_frame(path_ij)
#            frame_ij=np.expand_dims(frame_ij,0)
#            return model.predict(frame_ij)
#        pred_i=np.array([frame_helper(path_ij) for path_ij in frame_path])
#        out_i=out_path+'/'+action_i.split('/')[-1]
#        pred_i=np.squeeze(pred_i, axis=1)
#        np.savetxt(out_i, pred_i, delimiter=',' )

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

train_model("full")
#frames=read_frame_seqs("full")
#print( len(frames.split()[0]))