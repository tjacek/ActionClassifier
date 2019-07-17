import numpy as np
import keras.utils
from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D
from keras.preprocessing import sequence
from keras.models import Model,Sequential
from sklearn.metrics import classification_report
from sklearn import preprocessing
import data

def extract_features(in_path,out_path,n_epochs=10):
    extractor_model=make_extractor(in_path,n_epochs=n_epochs)
    data_dict=data.get_dataset(in_path,splited=False)
    max_len=max([x_i.shape[0] for x_i  in data_dict.values()])
    def seq_helper(x_i):
        x_i=sequence.pad_sequences(x_i.T,maxlen=max_len).T
        x_i=np.expand_dims(x_i,axis=0)
        return extractor_model.predict(x_i)
    feat_dict={name_i:seq_helper(data_i)
                  for name_i,data_i in data_dict.items()}
    save_feats(feat_dict,out_path)

def make_extractor(in_path,n_epochs=10):
    train_X,train_y,test_X,test_y,max_len,n_feats,n_cats=prepare_data(in_path)
    model=make_conv_lstm(n_cats,max_len,n_feats)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam')
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    extractor_model=Model(inputs=model.input,
                          outputs=model.get_layer("hidden").output)
    return extractor_model

def save_feats(feat_dict,out_path):
    text=""
    for name_i,x_i in feat_dict.items():
        sample_i=np.array2string(x_i,separator=",").replace('\n',"")
        text+=sample_i+'#'+name_i+'\n'
    text=text.replace("[","").replace("]","")
    file_str = open(out_path,'w')
    file_str.write(text)
    file_str.close()

def simple_exp(in_path,n_epochs=100):
    train_X,train_y,test_X,test_y,max_len,n_feats,n_cats=prepare_data(in_path)
    model=make_conv_lstm(n_cats,max_len,n_feats)
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
    #X=np.array([ preprocessing.scale(X_i) for X_i in X])
    return X,y

def make_base_lstm(n_cats,max_len,n_feats):
    model = Sequential()
    model.add(LSTM(units=30, return_sequences= True, input_shape=(max_len,n_feats)))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(LSTM(units=30,name="hidden"))
    model.add(Dense(units=n_cats,activation='softmax',name="hidden"))
    return model

def make_conv_lstm(n_cats,max_len,n_feats):
    input_layer = Input(shape=(max_len, n_feats))
    conv1 = Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same')(input_layer)
    lstm1 = LSTM(32, return_sequences=True)(conv1)
    lstm2 = LSTM(32,name="hidden")(lstm1)
    #lstm2=LSTM(32,return_sequences=False)(conv1)
    output_layer = Dense(units=n_cats, activation='softmax')(lstm2)
    return Model(inputs=input_layer, outputs=output_layer)

def max_seq_len(seqs):
	return max([seq_i.shape[0] for seq_i in seqs])

extract_features("mra","lstm.txt")
#model.summary()