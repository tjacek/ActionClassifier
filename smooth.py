import numpy as np
import keras.utils
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn.metrics import accuracy_score
import data,convnet

class GaussSmooth(object):
    def __init__(self,sigma=1,window=16):
        self.filtr = gaussian(window,sigma)
        self.filtr/=self.filtr.sum()

    def __call__(self,ts_i):
        return filters.convolve1d(ts_i,self.filtr)

def smooth_curve(in_path):
    data_dict=data.get_dataset(in_path,splited=False)
    smooth_data=smooth_dataset(data_dict,GaussSmooth())
    def conv_helper(smooth_data):
        train_X,train_y,test_X,test_y,params=convnet.prepare_data(smooth_data)
        model=convnet.make_conv(params)
        history=model.fit(train_X,train_y,epochs=100,batch_size=32)
        loss_i=history.history['loss'][-1]
        pred_y=model.predict_classes(test_X)
        true_y=np.argmax(test_y,axis=1)
        return loss_i, accuracy_score(true_y,pred_y)
    print(conv_helper(smooth_data))

def smooth_dataset(data_dict,smooth): 
    return { name_i:smooth_feature(sample_i,smooth)
                for name_i,sample_i in data_dict.items()}

def smooth_feature(sample_i,smooth):    
    return np.array([ smooth(feat_j)
                for feat_j in sample_i.T]).T

smooth_curve("mra")