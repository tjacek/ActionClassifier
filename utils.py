import numpy as np
from keras.models import load_model
from keras.models import Model
import data.feats,data.imgs

class TrainNN(object):
    def __init__(self,read,make_model):
        self.read=read
        self.make_model=make_model

    def __call__(self,in_path,nn_path,n_epochs=5):
        dataset=self.read(in_path)
        train,test=dataset.split()
        X,y,params=get_dataset(train)
        model=self.make_model(params)
        model.fit(X,y,epochs=n_epochs,batch_size=32)
        if(nn_path):
            model.save(nn_path)

class Extract(object):
    def __init__(self,read,name="hidden",preproc=None):
        self.read=read
        self.name=name
        self.preproc=preproc
        
    def __call__(self,in_path,nn_path,out_path):
        dataset=self.read(in_path)
        if(self.preproc):
            self.preproc(dataset)
        model=load_model(nn_path)
        extractor=Model(inputs=model.input,
                outputs=model.get_layer(self.name).output)
        extractor.summary()
        X,y=dataset.to_dataset()
        new_X=extractor.predict(X)
        names=dataset.names()
        feat_dict=data.feats.Feats()
        for i,name_i in enumerate(dataset.names()):
            feat_dict[name_i]=new_X[i] 
        feat_dict.save(out_path)

def check_model(nn_path):
    model=load_model(nn_path)
    model.summary()

def to_one_hot(y,n_cats=20):
    one_hot=np.zeros((len(y),n_cats))
    for i,y_i in enumerate(y):
        one_hot[i,y_i]=1
    return one_hot

def seq_stats(in_path):
    frames=data.imgs.read_frame_seqs(in_path,n_split=1)
    seqs_len=frames.seqs_len()
    print("%s,%s,%s" % (sum(seqs_len),min(seqs_len),max(seqs_len)))

if __name__=="__main__":
#    check_model("../action/ens/nn/0")
    seq_stats('../3DHOI/frames')