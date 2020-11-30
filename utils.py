#import numpy as np
#from keras.models import Model
#from sklearn.metrics import classification_report
#import data
from keras.models import load_model

def check_model(nn_path):
    model=load_model(nn_path)
    model.summary()

def simple_exp(in_path,prepare,make_model,n_epochs=10):
    model,(test_X,test_y)=train_model(in_path,prepare,make_model,n_epochs=n_epochs)
    raw_pred=model.predict(test_X,batch_size=32)
    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(test_y,axis=1)
    print(classification_report(test_y, pred_y,digits=4))

check_model("Data/MSR/binary/nn/nn0")