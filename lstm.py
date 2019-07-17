import numpy as np
import keras.utils
from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D
from keras.preprocessing import sequence
from keras.models import Model,Sequential
from sklearn.metrics import classification_report
from sklearn import preprocessing
import data

def simple_exp(in_path,n_epochs=100):
    train_X,train_y,test_X,test_y,max_len,n_feats,n_cats=prepare_data(in_path)
    model=make_base_lstm(n_cats,max_len,n_feats)
    model.summary()
    model.compile(loss='categorical_crossentropy',
    	          optimizer='adam')
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    raw_pred=model.predict(test_X,batch_size=32)
    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(test_y,axis=1)
    print(classification_report(test_y, pred_y,digits=4))

def prepare_data(in_path):
    (train_X,train_y),(test_X,test_y)=data.get_dataset(in_path)
    max_len=max(max_seq_len(train_X),max_seq_len(test_X))
    n_feats=train_X[0].shape[1]
    n_cats=np.unique(train_y).shape[0]
    (train_X,train_y),(test_X,test_y)=format_data(train_X,train_y,max_len),format_data(test_X,test_y,max_len)
    return train_X,train_y,test_X,test_y,max_len,n_feats,n_cats

def format_data(X,y,max_len):
    X=sequence.pad_sequences(X,maxlen=max_len)
    y=keras.utils.to_categorical(y)  
    X=np.array([ preprocessing.scale(X_i) for X_i in X])
    return X,y

def make_base_lstm(n_cats,max_len,n_feats):
    model = Sequential()
    model.add(LSTM(units=30, return_sequences= True, input_shape=(max_len,n_feats)))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(LSTM(units=30))
    model.add(Dense(units=n_cats,activation='softmax'))
    return model

def make_model(max_len,n_feats):
    input_layer = Input(shape=(max_len, n_feats))
    conv1 = Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same')(input_layer)
    lstm1 = LSTM(32, return_sequences=True)(conv1)
    output_layer = Dense(n_cats, activation='softmax')(lstm1)
    return Model(inputs=input_layer, outputs=output_layer)

def max_seq_len(seqs):
	return max([seq_i.shape[0] for seq_i in seqs])

simple_exp("mra")
#model.summary()