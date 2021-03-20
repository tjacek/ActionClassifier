from keras.layers import Layer
from keras.layers import Input,Dense,Conv1D, MaxPooling1D,LSTM
from keras.models import Model
import keras.backend as K
import files,utils,data.seqs,ens

class TS_LSTM(object):
    def __init__(self,atten=False,activ="relu"):
        self.activ=activ
        self.atten=atten

    def __call__(self,params):
        activ='relu'
        input_img=Input(shape=(params['ts_len'], params['n_feats']))
        n_kerns,kern_size,pool_size=[128,128],[8,8],[2,2]
        x=Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=self.activ,name='conv1')(input_img)
        x=MaxPooling1D(pool_size=pool_size[0],name='pool1')(x)
        x=Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=self.activ,name='conv2')(x)
        atten_name,lstm_name=("hidden",None) if(self.atten) else (None,"hidden")
        x=LSTM(64,return_sequences=self.atten,name=lstm_name,dropout=0.3,recurrent_dropout=0.2)(x)
        if(self.atten):
            x=SimpleAttention(name=atten_name)(x)
        outputs=Dense(params["n_cats"],activation='sigmoid',trainable=True)(x)
        model=Model(input_img,outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()
        return model

class SimpleAttention(Layer):
    def __init__(self,**kwargs):
        super(SimpleAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(SimpleAttention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(SimpleAttention,self).get_config()

def multi_exp(in_path,out_name,n_epochs=1000,size=64):
    out_path="%s/%s" % (in_path,out_name)
    train_dict={"lstm":TS_LSTM(atten=False),
                "atten":TS_LSTM(atten=True)}
    train_dict={name_i:utils.TrainNN(data.seqs.read_seqs,train_i,to_dataset)
                    for name_i,train_i in train_dict.items()}
    extract=utils.Extract(data.seqs.read_seqs)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ens.multimodel_ensemble(in_path,out_path,train_dict,
                            extract,arg_dict)

def ensemble1D(in_path,out_name,n_epochs=1000,size=64,atten=False):
    input_paths=files.top_files("%s/seqs" % in_path)
    out_path="%s/%s" % (in_path,out_name)
    files.make_dir(out_path)
    train,extract=get_train(atten)
    ensemble=ens.ts_ensemble(train,extract)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def single_exp(in_path,out_name="atten",atten=False):
    train,extract=get_train(atten)
    seq_path="%s/%s" % (in_path,"nn0")
    utils.single_exp_template(in_path,out_name,train,extract,seq_path)

def get_train(atten=False):
    read=data.seqs.read_seqs
    train=utils.TrainNN(read,TS_LSTM(atten),to_dataset)
    custom_layer={'SimpleAttention':SimpleAttention} if(atten) else None
    extract=utils.Extract(read,custom_layer=custom_layer)
    return train,extract

def to_dataset(seqs):
    X,y=seqs.to_dataset()
    n_cats=max(y)+1
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],'n_cats':n_cats}
    return X,y,params

in_path="../dtw_paper/MSR/binary"
multi_exp(in_path,"1D_CNN",n_epochs=1000,size=64)