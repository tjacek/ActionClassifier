import numpy as np
import keras.models
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import gc
import data.feats,data.imgs,data.seqs
import files,spline

class TrainNN(object):
    def __init__(self,read,make_model,to_dataset):
        self.read=read      
        self.make_model=make_model
        self.to_dataset=to_dataset

    def __call__(self,in_path,nn_path,n_epochs=5):
        dataset=self.read(in_path)
        train,test=dataset.split()
        X,y,params=self.to_dataset(train)
        y=to_one_hot(y,params["n_cats"])
        model=self.make_model(params)
        model.fit(X,y,epochs=n_epochs,batch_size=32)
        if(nn_path):
            model.save(nn_path)

class Extract(object):
    def __init__(self,read,name="hidden",preproc=None,custom_layer=None):
        self.read=read
        self.name=name
        self.preproc=preproc
        self.custom_layer=custom_layer
    
    def get_model(self,nn_path):
        if(self.custom_layer):
            return keras.models.load_model(nn_path,
                        custom_objects=self.custom_layer)
        return load_model(nn_path)

    def __call__(self,in_path,nn_path,out_path):
#        K.clear_session()
#        gc.collect()
        dataset=self.read(in_path)
        if(self.preproc):
            self.preproc(dataset)
#        model=load_model(nn_path)
        model=self.get_model(nn_path)
        extractor=Model(inputs=model.input,
                outputs=model.get_layer(self.name).output)
        extractor.summary()
        X,y=dataset.to_dataset()
        names=dataset.names()
        feat_dict=data.feats.Feats()
        for i,name_i in enumerate(dataset.names()):
            x_i=np.expand_dims(X[i],axis=0)#extractor.predict(X[i])
            feat_dict[name_i]=extractor.predict(x_i)
        feat_dict.save(out_path)

class ExtractSeqs(object):
    def __init__(self,read=None,name="hidden",preproc=None):
        if(read is None):
            read=data.imgs.read_frame_seqs
        self.read=read
        self.name=name
        self.preproc=preproc
        
    def __call__(self,in_path,nn_path,out_path):
        gc.collect()
        dataset=self.read(in_path)#,n_split=1)
        if(self.preproc):
            self.preproc(dataset)
        model=load_model(nn_path)
        extractor=Model(inputs=model.input,
                outputs=model.get_layer(self.name).output)
        extractor.summary()
        names=dataset.names()
        feat_seqs=data.seqs.Seqs()
        for i,name_i in enumerate(dataset.names()):
            x_i=np.array(dataset[name_i])
            feat_seqs[name_i]= extractor.predict(x_i)
        feat_seqs.save(out_path)

def single_exp_template(in_path,out_name,train,extract,seq_path=None):
    paths=files.prepare_dirs(in_path,out_name,["spline","nn","feats"])
    if(seq_path):
        paths["seqs"]=seq_path#"%s/%s" % (paths["seqs"],"nn0")
    print(paths)
    spline.upsample(paths['seqs'],paths['spline'],size=64)
    train(paths["spline"],paths["nn"])
    extract(paths["spline"],paths["nn"],paths["feats"])


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
    stats=(sum(seqs_len),min(seqs_len),max(seqs_len))
    print("n_frames:%s\nmin_len:%s\nmax_len:%s\n" % stats)
    stats=(len(seqs_len),frames.n_persons())
    print("n_seqs:%s\nn_persons:%s\n" % stats)

def mean_action_size(in_path):
    frames=data.imgs.read_frame_seqs(in_path,n_split=1)
    cats={}
    for name_i,seq_i in frames.items():
        cat_i=name_i.get_cat()
        if( name_i.get_person()%2==1):
            if(not cat_i in cats ):
                cats[cat_i]=0
            cats[cat_i]+=len(seq_i)
    print(cats)
    values=list(cats.values())
    print(np.median(values))
    print(sum(values))

if __name__=="__main__":
#    check_model("../action/ens/nn/0")
    mean_action_size('../../common/frames/full')