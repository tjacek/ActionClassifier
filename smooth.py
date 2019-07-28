import numpy as np
import keras.utils
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import data,convnet

class GaussSmooth(object):
    def __init__(self,sigma=1,window=16):
        self.filtr = gaussian(window,sigma)
        self.filtr/=self.filtr.sum()

    def __call__(self,ts_i):
        return filters.convolve1d(ts_i,self.filtr)

def smooth_plot(in_path,n_epochs=100):
    data_dict=data.get_dataset(in_path,splited=False)
    loss,acc=[],[]
    sigma=np.arange(1,10)
    for i in sigma:    
        print("\n \n sigma %d" % i)
        smooth_i=smooth_dataset(data_dict,GaussSmooth(i))
        loss_i,acc_i= single_exp(smooth_i,n_epochs) #conv_helper(smooth_i,n_epochs)
        print("acc %f std %f" %acc_i)
        loss.append(loss_i)
        acc.append(acc_i)
    plot_curve(sigma,loss)
    plot_curve(sigma,acc)
    
def plot_curve(x,y):
    y=np.array(y)
    print(y[:,0])
    print(y[:,1])
    plt.errorbar(x,y[:,0],y[:,1])
    plt.show()

def single_exp(smooth_i,n_epochs):
    result=[conv_helper(smooth_i,n_epochs) 
                for i in range(10)]
    mean_i=np.mean(result,axis=0)
    std_i=np.std(result,axis=0)
    return (mean_i[0],std_i[0]),(mean_i[1],std_i[1])

def conv_helper(smooth_data,n_epochs=10):
    train_X,train_y,test_X,test_y,params=convnet.prepare_data(smooth_data)
    model=convnet.make_conv(params)
    history=model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    loss_i=history.history['loss'][-1]
    pred_y=model.predict_classes(test_X)
    true_y=np.argmax(test_y,axis=1)
    return loss_i, accuracy_score(true_y,pred_y)

def smooth_dataset(data_dict,smooth): 
    return { name_i:smooth_feature(sample_i,smooth)
                for name_i,sample_i in data_dict.items()}

def smooth_feature(sample_i,smooth):    
    return np.array([ smooth(feat_j)
                for feat_j in sample_i.T]).T

smooth_plot("mra")