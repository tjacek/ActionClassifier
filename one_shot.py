import keras,keras.backend as K
import keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,Lambda#GlobalAveragePooling1D
from keras import regularizers
from keras.models import load_model
import numpy as np
import feats

class DtwModel(object):
    def __init__(self,first=100,last=64):
        self.first=first
        self.last=last

    def __call__(self,model):
        activ='relu'
        model.add(Dense(self.first, activation=activ))
        model.add(Dense(self.last, activation=activ,name='hidden',
            kernel_regularizer=regularizers.l1(0.01)))
        return model

def dtw_one_shot(in_path,out_path=None,n_epochs=5):
    dtw_feats=feats.read_feats(in_path)
    train,test=dtw_feats.split()
    X,y=to_sim_dataset(train)
    params={'input_shape':(dtw_feats.dim(),)}
    siamese_net,extractor=build_siamese(params,DtwModel())
    siamese_net.fit(X,y,epochs=n_epochs,batch_size=64)
    if(out_path):
        extractor.save(out_path)

def dtw_extract(in_path,nn_path,out_path):
    dtw_feats=feats.read_feats(in_path)
    dtw_feats.norm()
    extractor=load_model(nn_path)
    def helper(x_i):
        x_i=np.expand_dims(x_i,0)
        result_i= extractor.predict(x_i)
#        raise Exception(result_i.shape)
        return result_i
    dtw_feats=dtw_feats.transform(helper)
    dtw_feats.save(out_path)

def to_sim_dataset(dtw_feats):
    dtw_feats.norm()
    pairs=all_pairs(dtw_feats.names())
    X,y=[],[]
    for name_i,name_j in pairs:
        X.append((dtw_feats[name_i],dtw_feats[name_j]))
        y.append(int(name_i.split('_')[0]==name_j.split('_')[0]))
    print(y)
    X=np.array(X)
    X=[X[:,0],X[:,1]]
#    y=keras.utils.to_categorical(y)
    return X,y

def all_pairs(names):
    pairs=[]
    for i,name_i in enumerate(names):
        for name_j in names[i+1:]:
            pairs.append((name_i,name_j))
    return pairs

def build_siamese(params,make_model):
    input_shape=params["input_shape"]
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    make_model(model)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    prediction,loss=contr_loss(encoded_l,encoded_r)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    optimizer = keras.optimizers.Adam(lr = 0.00006)
    siamese_net.compile(loss=loss,optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor

def contr_loss(encoded_l,encoded_r):
    L2_layer = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)
    return L2_layer([encoded_l, encoded_r]),contrastive_loss

def contrastive_loss(y_true, y_pred):
    margin = 50
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

if __name__ == "__main__":
    in_path=["dtw/maxz/feats","dtw/corl/feats","dtw/skew/feats"]
    dtw_one_shot(in_path,"dtw_nn",100)
    dtw_extract(in_path,"dtw_nn","sim_feats")