import numpy as np

def get_sample(seq_i):
    if(type(seq_i)==list):
        size=len(seq_i)
    else:   
        size=seq_i.shape[0]
    dist_i=get_dist(size)
    def sample_helper(n):   
        return np.random.choice(np.arange(size),n,p=dist_i)
    return sample_helper

def get_dist(n):
    inc,dec=np.arange(n),np.flip(np.arange(n))
    dist=np.amin(np.array([inc,dec]),axis=0)
    dist=dist.astype(float)
    dist=dist**2
    if(np.sum(dist)==0):
        dist.fill(1.0)
    dist/=np.sum(dist)
    return dist