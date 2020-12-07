from keras.models import load_model

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
    def __init__(self,read):
        self.read=read
        
    def __call__(self,in_path,nn_path,out_path):
        dataset=self.read(in_path)
        model=load_model(nn_path)
        extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
        X,y,params=get_dataset(dataset)
        new_X=model.predict(X)
        names=dataset.names()
        feat_dict={name_i:new_X[i] 
                for i,name_i in enumerate(dataset.names())}
        data.save_feats(feat_dict,out_path)

def check_model(nn_path):
    model=load_model(nn_path)
    model.summary()