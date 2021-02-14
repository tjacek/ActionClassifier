import numpy as np
import scipy.signal
from scipy.interpolate import CubicSpline
import files,data.seqs

class SplineUpsampling(object):
    def __init__(self,new_size=128):
        self.name="spline"
        self.new_size=new_size

    def __call__(self,feat_i):
        print(feat_i.shape)
        old_size=feat_i.shape[0]
        old_x=np.arange(old_size)
        old_x=old_x.astype(float)  
        if(self.new_size):
            step=float(self.new_size)/float(old_size)
            old_x*=step     
            cs=CubicSpline(old_x,feat_i)
            new_size=np.arange(self.new_size)  
            return cs(new_size)
        else:
            cs=CubicSpline(old_x,feat_i)
            return cs(old_x)

def ens_upsample(in_path,out_path,size=64):
    files.ens_template(in_path,out_path,upsample)

def upsample(in_path,out_path,size=64):
    print(in_path)
    seq_dict=data.seqs.read_seqs(in_path)
    spline=SplineUpsampling(size)
    seq_dict={ name_i:spline(seq_i) for name_i,seq_i in seq_dict.items()
                    if(seq_i.shape[0]>1)}
    seq_dict=data.seqs.Seqs(seq_dict)
    seq_dict.save(out_path)

if __name__ == "__main__":
    ens_upsample("Data/MSR/seqs","Data/MSR/spline")