import numpy as np
import os.path
import seqs,files,spline,data
import convnet

def unified_exp(in_path,n_epochs=1000):
    united_path="%s/united" % in_path
    files.make_dir(united_path)
    paths=files.get_paths(united_path,["spline","nn","feats"])
    paths["seqs"]="%s/seqs" % in_path
    spline.ens_upsample(paths["seqs"],paths["spline"],size=64)
    convnet.train_nn(paths["spline"],paths["nn"],n_epochs,
                read=read_unified)
    convnet.extract(paths["spline"],paths["nn"],paths["feats"],
                    read=read_unified)

def read_unified(in_path):
    all_seqs=[seqs.read_seqs(path_i) 
        for path_i in files.top_files(in_path)]
    names=all_seqs[0].names()
    unified=seqs.Seqs()
    for name_i in names:
        seq_i=[ dict_j[name_i]  for dict_j in all_seqs]
        unified[name_i]=np.concatenate(seq_i,axis=1)
    unified=data.norm_local(unified)
    return unified

unified_exp("Data/3DHOI")