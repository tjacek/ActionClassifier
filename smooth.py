import numpy as np
from scipy.signal import gaussian
from scipy.ndimage import filters
import data

def smooth_curve(in_path):
    data_dict=data.get_dataset(in_path,splited=False)
    for name_i,data_i in data_dict.items():
        print(np.mean(gauss_smooth(data_i) - data_i))
        #gauss_smooth(train_X)
 
def gauss_smooth(data_i,sigma=1):
    filtr = gaussian(16,sigma)
    filtr/=filtr.sum()
    return np.array([filters.convolve1d(feat_j,filtr)
                        for feat_j in data_i.T]).T

smooth_curve("mra")