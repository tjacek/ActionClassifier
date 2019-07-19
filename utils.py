import numpy as np
from keras.models import Model
from sklearn.metrics import classification_report
import data

def simple_exp(in_path,prepare,make_model,n_epochs=10):
    model=train_model(in_path,prepare,make_model,n_epochs=n_epochs)
    raw_pred=model.predict(test_X,batch_size=32)
    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(test_y,axis=1)
    print(classification_report(test_y, pred_y,digits=4))

def extract_features(in_path,out_path,prepare,
	                        make_model,extraction,n_epochs=100):
    model=train_model(in_path,prepare,make_model,n_epochs=n_epochs)
    extractor_model=make_extractor(model)
    data_dict=data.get_dataset(in_path,splited=False)
    feat_dict=extraction(data_dict,extractor_model)
    data.save_feats(feat_dict,out_path)

def train_model(in_path,prepare,make_model,n_epochs=100):
    train_X,train_y,test_X,test_y,params=prepare(in_path)
    model=make_model(params)
    model.summary()
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    return model

def make_extractor(model):
    return Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)