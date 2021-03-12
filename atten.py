from keras.layers import Layer
from keras.layers import Input,Dense,Conv1D, MaxPooling1D,LSTM
from keras.models import Model
import keras.backend as K
import files,utils,data.seqs,spline,ens

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

def ensemble1D(in_path,out_name,n_epochs=1000,size=64):
    input_paths=files.top_files("%s/seqs" % in_path)
    out_path="%s/%s" % (in_path,out_name)
    files.make_dir(out_path)
    train=get_train()
    extract=utils.Extract(data.seqs.read_seqs,custom_layer={'SimpleAttention':SimpleAttention})
    ensemble=ens.ts_ensemble(train,extract)
    arg_dict={'size':size,'n_epochs':n_epochs}
    ensemble(input_paths,out_path, arg_dict)

def single_exp(in_path,out_name="atten"):
    train=get_train()
    extract=utils.Extract(data.seqs.read_seqs)
    seq_path="%s/%s" % (in_path,"nn0")
    utils.single_exp_template(in_path,out_name,train,extract,seq_path)

def get_train():
    read=data.seqs.read_seqs
    return utils.TrainNN(read,make_lstm,to_dataset)

def to_dataset(seqs):
    X,y=seqs.to_dataset()
    n_cats=max(y)+1
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],'n_cats':n_cats}
    return X,y,params

def make_lstm(params):
    activ='relu'
    input_img=Input(shape=(params['ts_len'], params['n_feats']))
    n_kerns,kern_size,pool_size=[128,128],[8,8],[2,2]
    x=Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=activ,name='conv1')(input_img)
    x=MaxPooling1D(pool_size=pool_size[0],name='pool1')(x)
    x=Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=activ,name='conv2')(x)
#    x=MaxPooling1D(pool_size=pool_size[1],name='pool2')(x)
    att_in=LSTM(64,return_sequences=True,dropout=0.3,recurrent_dropout=0.2)(x)
    att_out=SimpleAttention(name="hidden")(att_in)
    outputs=Dense(params["n_cats"],activation='sigmoid',trainable=True)(att_out)
    model=Model(input_img,outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model

in_path="../dtw_paper/MSR/binary"
#single_exp(in_path)
ensemble1D(in_path,"atten",n_epochs=10,size=64)