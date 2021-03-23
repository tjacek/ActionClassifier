from keras.layers import Input,Dense
import deep,data.seqs,data.feats

class TS_REG(object):
    def __init__(self,optim_alg=None):
        if(optim_alg is None):
            optim_alg=deep.RMS()
        self.optim_alg=optim_alg

    def __call__(self,params):
        input_img=Input(shape=(params['ts_len'], params['n_feats']))
        n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
        x=deep.add_conv_layer(input_img,n_kerns,kern_size,
                            pool_size,activ=self.activ,one_dim=True)
        x=deep.full_layer(x)
        x=Dense(units=400,activation='relu')(x)
        model = Model(input_img, x)
        model.compile(loss='mean_squared_error',
              optimizer=self.optim_alg())
        model.summary()
        return model

def simple_exp(x_path,y_path,n_epochs=1000):
    seq_dict=data.seqs.read_seqs(x_path)
    feat_dict=data.feats.read_feats(y_path)
    X=seq_dict.to_dataset()[0]
    y=feat_dict.to_dataset()[0]
    params={'ts_len':64,'n_feats':100,'n_cats':400}
    return X,y,params
    make_model=TS_REG() 
    make_model(params)
#    print(len(X))
#    print(y.shape)

x_path="../dtw_paper/MSR/binary/seqs/nn0"
y_path="../dtw_paper/MSR/binary/stats/feats/nn0"
simple_exp(x_path,y_path,n_epochs=1000)