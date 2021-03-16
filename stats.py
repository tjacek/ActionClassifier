import numpy as np
import scipy.stats
import ens,data.seqs,files

class Stats(object):
    def __init__(self,fun):
        self.fun=fun

    def __call__(self,in_path,out_path):
        seq_dict=data.seqs.read_seqs(in_path)
        def helper(ts_i):
            if(np.all(ts_i==0)):
                return np.zeros((len(self.fun)))
            feats_i=np.array([fun_j(ts_i) for fun_j in self.fun])
            return feats_i
        feat_dict=seq_dict.to_feats(helper,single=True)
        feat_dict.save(out_path)

    def ens(self,in_path,out_path):
        ens.ens_template(in_path,out_path,self)

def get_base_stats():
    return Stats([np.mean,np.std,scipy.stats.skew,time_corl])

def get_simple_stats():
    return Stats([np.mean,np.std,time_corl])

def get_extended_stats():
    return Stats([np.mean,np.std,scipy.stats.skew,
                    time_corl,scipy.stats.kurtosis])

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)
    return scipy.stats.pearsonr(x_i,feat_i)[0]

def basic_stats(data_i):
    return [fun(data_i) for fun in [np.amax,np.amin,np.mean,np.median]]

class RobustJarqueBera(object):
    def __init__(self,):
        self.C=np.sqrt(np.pi/2)
        self.C1=6.0
        self.C2=64.0

    def j_n(self,ts_i):
        sum_i=np.sum(np.abs(ts_i-np.median(ts_i)))
        return (self.C/ts_i.size)*sum_i

    def __call__(self,ts_i):
        u_3=scipy.stats.moment(ts_i, moment=3)
        u_4=scipy.stats.moment(ts_i, moment=4)
        j=self.j_n(ts_i)
        n=ts_i.shape[0]
        rjb=(n/self.C1)*(u_3/(j**3))**2 +(n/self.C2)*( u_4/(j**4) -3)**2
        p = 1 - scipy.stats.distributions.chi2.cdf(rjb, 2)
        return p

def check_norm(in_path,single=False,fun=None):
    if(fun is None):
        fun=shapiro_test
    if(not single):
        results=[ check_norm(path_i,True,fun) 
                for path_i in files.top_files(in_path)]
        return basic_stats(results)
    seq_dict=data.seqs.read_seqs(in_path)
    def norm_test(ts_i):
        if(np.all(ts_i)==0):
            return -1.0
        return fun(ts_i)<0.05
    feat_dict= seq_dict.to_feats(norm_test,single=True)
    X,y=feat_dict.to_dataset()
    size,no_pass=np.product(X[X>=0].shape),np.sum(X[X>=0])
    print(size,no_pass)
    return (no_pass/size)

def shapiro_test(ts_i):
    return float(scipy.stats.shapiro(ts_i)[1])

def check_feature(in_path):
    all_ts=[]
    for path_i in files.top_files(in_path):
        seq_dict_i=data.seqs.read_seqs(path_i)
        feat_dict_i= seq_dict_i.to_feats(scipy.stats.kurtosis,single=True)
        X,y=feat_dict_i.to_dataset()
        all_ts.append(X.flatten())
    all_ts=np.array(all_ts).flatten()
    print(basic_stats(all_ts))

if __name__ == "__main__":
    norm_test=RobustJarqueBera()
    stats=check_norm("../dtw_paper/MSR/binary/seqs",fun=norm_test)
    print(stats)

#    dataset="MHAD"
#    in_path="../dtw_paper/%s/binary/seqs" % dataset
#    out_path="../dtw_paper/%s/binary/stats_mod/ext" % dataset
#    stats=get_extended_stats()
#    stats.ens(in_path,out_path)