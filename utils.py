import numpy as np
import keras.models
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import gc
import data.feats,data.imgs,data.seqs
import files,spline

class TrainNN(object):
    def __init__(self,read,make_model,to_dataset,batch_size=32):
        self.read=read      
        self.make_model=make_model
        self.to_dataset=to_dataset
        self.batch_size=batch_size

    def __call__(self,in_path,nn_path,n_epochs=5):
        dataset=self.read(in_path)
        train,test=dataset.split()
        X,y,params=self.to_dataset(train)
        if("n_cats" in params ):
            y=to_one_hot(y,params["n_cats"])
        model=self.make_model(params)
        model.fit(X,y,epochs=n_epochs,batch_size=self.batch_size)
        if(nn_path):
            model.save(nn_path)

class Extract(object):
    def __init__(self,read,name="hidden",preproc=None,custom_layer=None):
        self.read=read
        self.name=name
        self.preproc=preproc
        self.custom_layer=custom_layer

    def __call__(self,in_path,nn_path,out_path):
        K.clear_session()
        gc.collect()
        dataset=self.read(in_path)
        if(self.preproc):
            self.preproc(dataset)
        model=load_model(nn_path,custom_objects=self.custom_layer)
        if(self.name):
            extractor=Model(inputs=model.input,
                outputs=model.get_layer(self.name).output)
        else:
            extractor=model
        extractor.summary()
        X,y=dataset.to_dataset()
        names=dataset.names()
        feat_dict=data.feats.Feats()
        for i,name_i in enumerate(dataset.names()):
            x_i=np.expand_dims(X[i],axis=0)
            feat_dict[name_i]=extractor.predict(x_i)
            if(not self.name):
                feat_dict[name_i]=feat_dict[name_i].T
        if(out_path):
            feat_dict.save(out_path)
        return feat_dict

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

def single_exp_template(in_path,out_path,train,extract,
                            size=64,n_epochs=1000):
    paths=files.prepare_dirs(out_path,["spline","nn","feats"])
    paths['seqs']=in_path
    spline.upsample(paths['seqs'],paths['spline'],size)
    train(paths["spline"],paths["nn"],n_epochs=n_epochs)
    extract(paths["spline"],paths["nn"],paths["feats"])

def check_model(nn_path):
    model=load_model(nn_path)
    model.summary()

def to_one_hot(y,n_cats=20):
    one_hot=np.zeros((len(y),n_cats))
    for i,y_i in enumerate(y):
        one_hot[i,y_i]=1
    return one_hot