import os,os.path,re
import numpy as np

def get_dataset(in_path,splited=True):
    data_dict=read_data(in_path)
    data_dict=norm_seqs(data_dict)
    if(splited):
        train,test=split(data_dict)
        return to_array(train),to_array(test)
    return data_dict

def read_data(in_path):
    if(multiple_dataset(in_path)):
        datasets=[read_single(path_i) for path_i in top_files(in_path)]
        names=datasets[0].keys()
        def name_helper(name_i):
            values=[dataset_i[name_i] for dataset_i in datasets]
            seq_len= min([value_i.shape[0] for value_i in values])
            values=[value_i[:seq_len] for value_i in values]
            data_i=np.concatenate(values,axis=1)
            return name_i,data_i
        return dict([ name_helper(name_i) for name_i in names])
    else:
        return read_single(in_path)

def read_single(in_path):
    all_paths=bottom_files(in_path)
    return dict([read_seq(path_i) for path_i in all_paths])

def read_seq(path_i):
    name_i=path_i.split('/')[-1]
    name_i=clean(name_i)#re.sub(r'[a-z]','',name_i.strip())
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

def top_files(path):
    return [ path+'/'+file_i for file_i in os.listdir(path)]
    	
def bottom_files(path,full_paths=True):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            for filename_i in filenames:
                path_i= root+'/'+filename_i if(full_paths) else filename_i
                all_paths.append(path_i)
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def show_seq_len(data_dict):
    for name_i,data_i in data_dict.items():
        print((name_i+" %d %d") % data_i.shape )

def multiple_dataset(in_path):
    names= bottom_files(in_path,False)
    if(not names):
        raise Exception("No data at: "+in_path)
    names=[clean(name_i) for name_i in names]
    first=names[0]
    for name_i in names[1:]:
        if(first==name_i):
            return True
    return False

def norm_seqs(data_dict):
    all_X=np.concatenate(data_dict.values(),axis=0)
    all_mean=np.mean(all_X,axis=0)
    all_std=np.std(all_X,axis=0)
    for name_i,data_i in data_dict.items():
        data_dict[name_i]=(data_i-all_mean)/all_std
    return data_dict

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def save_feats(feat_dict,out_path):
    text=""
    for name_i,x_i in feat_dict.items():
        sample_i=np.array2string(x_i,separator=",").replace('\n',"")
        text+=sample_i+'#'+name_i+'\n'
    text=text.replace("[","").replace("]","")
    file_str = open(out_path,'w')
    file_str.write(text)
    file_str.close()

def clean(name_i):
    return "_".join([str(int(n_i)) for n_i in re.findall(r'\d+',name_i)])