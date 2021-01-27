import numpy as np
import keras,keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Lambda

class SimTrain(object):
    def __init__(self,get_model,get_cat,read=None):
        self.get_model=get_model
        self.get_cat=get_cat
        self.read=read

    def __call__(self,data_dict,out_path=None,n_epochs=5,params=None):
        if(type(data_dict)==str):
            data_dict=self.read(data_dict)
        train,test=data_dict.split()
        X,y=pairs_dataset(train,self.get_cat)
        if(not params):
            params={'input_shape':(data_dict.dim(),)}
        siamese_net,extractor=build_siamese(params,self.get_model)
        siamese_net.fit(X,y,epochs=n_epochs,batch_size=64)
        if(out_path):
            extractor.save(out_path)

def pairs_dataset(data_dict,get_cat):
    pairs=all_pairs(data_dict.names())
    X,y=[],[]
    for name_i,name_j in pairs:
        cat_i=get_cat(name_i,name_j)
        if(not (cat_i is None)):
            X.append((data_dict[name_i],data_dict[name_j]))
            y.append(cat_i  )
    X=np.array(X)
    X=[X[:,0],X[:,1]]
    return X,y

def all_pairs(names):
    names=list(names)
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