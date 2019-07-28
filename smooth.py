import numpy as np
from scipy.signal import gaussian
from scipy.ndimage import filters
import data

class GaussSmooth(object):
    def __init__(self,sigma=1,window=16):
        self.filtr = gaussian(window,sigma)
        self.filtr/=self.filtr.sum()

    def __call__(self,ts_i):
        return filters.convolve1d(ts_i,self.filtr)

def smooth_curve(in_path):
    data_dict=data.get_dataset(in_path,splited=False)
    new_data_dict=smooth_dataset(data_dict,GaussSmooth())
 
def smooth_dataset(data_dict,smooth): 
    return { name_i:smooth_feature(sample_i,smooth)
                for name_i,sample_i in data_dict.items()}

def smooth_feature(sample_i,smooth):    
    return np.array([ smooth(feat_j)
                for feat_j in sample_i.T]).T

smooth_curve("mra")