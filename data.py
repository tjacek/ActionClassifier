import os,re
import numpy as np

def get_dataset(in_path):
    data_dict=read_data(in_path)
    train,test=split(data_dict)
    return to_array(train),to_array(test)

def read_data(in_path):
    all_paths=bottom_files(in_path)
    return dict([read_seq(path_i) for path_i in all_paths])

def read_seq(path_i):
    name_i=path_i.split('/')[-1]
    name_i=re.sub(r'[a-z]','',name_i.strip())
    data_i=np.loadtxt(path_i,dtype=float,delimiter=",")
    return (name_i,data_i)

def split(data_dict):
    train,test={},{}
    for name_i in data_dict.keys():
        person_i=int(name_i.split('_')[1])
        if(person_i%2==1):
            train[name_i]=data_dict[name_i]
        else:
            test[name_i]=data_dict[name_i]
    return train,test

def to_array(data_dict):
    names=sorted(data_dict.keys(),key=natural_keys) 
    X=[data_dict[name_i] for name_i in names]
    y=[ int(name_i.split('_')[0])-1 for name_i in names]
    return X,y
    	
def bottom_files(path):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            paths=[ root+'/'+filename_i 
                for filename_i in filenames]
            all_paths+=paths
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def show_seq_len(data_dict):
    for name_i,data_i in data_dict.items():
        print((name_i+" %d %d") % data_i.shape )