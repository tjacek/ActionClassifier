#import shutil,os
import sys
sys.path.append("..")
import json
import data.imgs,files,lstm

def rename_exp(in_path,rename_path,out_path):
    seqs_dict=data.imgs.read_frame_seqs(in_path)
    seqs_dict.scale()
    rename_dict=rename_dicts(seqs_dict,rename_path)
    pair=(in_path,rename_dict)
    lstm.binary_lstm(pair,out_path,n_epochs=5,seq_len=30,n_cats=12)

def rename_dicts(seqs_dict,rename_path):
    rename=read_rename(rename_path)
    print(rename.keys())
    rename_seqs=data.imgs.FrameSeqs()
    for name_i,rename_i in rename.items():
        rename_seqs[rename_i]=seqs_dict[name_i]
    return rename_seqs

def read_rename(path):
    rename= json.load(open("%s.json" % path))
    return { files.Name(name_i):files.Name(rename_i) 
                for name_i,rename_i in rename.items()}

#def add_cat(name_i):
#    raw=name_i.split("_")
#    cat=str(int(raw[0])+1)
#    raw=[cat]+raw[1:]
#    return "_".join(raw)

in_path="../../raw_3DHOI/3DHOI/frames"
rename_exp(in_path,"rename","test")