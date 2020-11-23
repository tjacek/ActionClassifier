import os,os.path,re
import numpy as np

def show_seq_len(data_dict):
    for name_i,data_i in data_dict.items():
        print((name_i+" %d %d") % data_i.shape )

def norm_seqs(data_dict):
    all_X=np.concatenate(data_dict.values(),axis=0)
    all_mean=np.mean(all_X,axis=0)
    all_std=np.std(all_X,axis=0)
    for name_i,data_i in data_dict.items():
        data_dict[name_i]=(data_i-all_mean)/all_std
    return data_dict

def norm_local(data_dict):
    for name_i,data_i in data_dict.items():
        mean_i=np.mean(data_i,axis=0)
        std_i=np.std(data_i,axis=0)
        data_dict[name_i]=(data_i-mean_i)/std_i
    return data_dict

def save_feats(feat_dict,out_path):
    lines=[]
    for name_j,feat_j in feat_dict.items():
        feat_j=np.nan_to_num(feat_j, copy=False)
        line_i=np.array2string(feat_j,separator=",")
        line_i=line_i.replace('\n',"")+'#'+name_j
        lines.append(line_i)
    feat_txt='\n'.join(lines)
    feat_txt=feat_txt.replace('[','').replace(']','')
    file_str = open(out_path,'w')
    file_str.write(feat_txt)
    file_str.close()