import numpy as np
import scipy.stats
import ens,data.seqs

def ens_stast(in_path,out_path):
	ens.ens_template(in_path,out_path,extract_feats)

def extract_feats(in_path,out_path):
	seq_dict=data.seqs.read_seqs(in_path)
	feat_dict=seq_dict.to_feats(feat_vector)
	feat_dict.save(out_path)

def feat_vector(seq_j):
    feats=[]
    for ts_k in seq_j.T:
    	feats+=EBTF(ts_k)
    return np.array(feats)

def EBTF(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0,0.0]
    return [np.mean(feat_i),np.std(feat_i),
    	                scipy.stats.skew(feat_i),time_corl(feat_i)]

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)
    return scipy.stats.pearsonr(x_i,feat_i)[0]

ens_stast("../ens/seqs","../ens/feats" )