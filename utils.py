import numpy as np
from sklearn.metrics import classification_report

def simple_exp(in_path,prepare,make_model,n_epochs=100):
    train_X,train_y,test_X,test_y,params=prepare(in_path)
    model=make_model(params)
    model.summary()
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    raw_pred=model.predict(test_X,batch_size=32)
    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(test_y,axis=1)
    print(classification_report(test_y, pred_y,digits=4))