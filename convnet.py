import os.path
import keras,keras.backend as K,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
from keras.models import load_model
import files,spline,seqs,data

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

def get_train():
    return TrainNN(seqs.read_seqs,clf_model)

def extract(in_path,nn_path,out_path,
            read=seqs.read_seqs):
    dataset=read(in_path)
    model=load_model(nn_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    X,y,params=get_dataset(dataset)
    new_X=model.predict(X)
    names=dataset.names()
    feat_dict={name_i:new_X[i] 
                for i,name_i in enumerate(dataset.names())}
    data.save_feats(feat_dict,out_path)

def get_dataset(seqs):
    X,y=seqs.to_dataset()
    y=keras.utils.to_categorical(y)
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],
                'n_cats':y.shape[1]}
    return X,y,params

def clf_model(params):
    x,input_img=basic_model(params)
    x=Dropout(0.5)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model

def basic_model(params):
    activ='relu'
    input_img=Input(shape=(params['ts_len'], params['n_feats']))
    n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
    x=Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=activ,name='conv1')(input_img)
    x=MaxPooling1D(pool_size=pool_size[0],name='pool1')(x)
    x=Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=activ,name='conv2')(x)
    x=MaxPooling1D(pool_size=pool_size[1],name='pool2')(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    return x,input_img

def basic_exp(in_path,n_epochs=1000):
    path,name=os.path.split(in_path)
    basic_path="%s/basic" % path
    files.make_dir(basic_path)
    paths=files.get_paths(basic_path,["spline","nn","feats"])
    paths["seqs"]=in_path
    train_nn=TrainNN()
    spline.upsample(paths['seqs'],paths['spline'],size=64)
    train_nn(paths["spline"],paths["nn"],n_epochs)
    extract(paths["spline"],paths["nn"],paths["feats"])

if __name__ == "__main__":
    basic_exp("Data/MSR/common/seqs")