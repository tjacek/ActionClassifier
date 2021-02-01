import numpy as np
import cv2
from keras.models import load_model
import files,data.seqs

class FrameSeqs(dict):
    def __init__(self, args=[]):
        super(FrameSeqs, self).__init__(args)

    def n_cats(self):
        cats=[ name_i.get_cat() for name_i in self.keys()]
        return max(cats)+1

    def n_frames(self):
        n=0
        for seq_i in self.values():
            n+=len(seq_i)
        return n

    def names(self):
        return sorted(self.keys(),key=files.natural_keys) 

    def dims(self):
        return list(self.values())[0][0].shape

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return FrameSeqs(train),FrameSeqs(test)

    def scale(self,dims=(64,64),new=False):
        def helper(img_j):
            img_j=cv2.resize(img_j,dsize=dims,interpolation=cv2.INTER_CUBIC)
            return np.expand_dims(img_j,axis=-1)
        return self.transform(helper,new,single=True)

    def transform(self,fun,new=False,single=True):
        frame_dict= FrameSeqs() if(new) else self
        for name_i,seq_i in self.items():
            if(single):
                frame_dict[name_i]=[fun(img_j)
                            for img_j in seq_i]
            else:
                frame_dict[name_i]=fun(seq_i)
        return frame_dict

    def to_dataset(self):
        names=self.names()
        X=[ np.array(self[name_i]) for name_i in names]
        y=[name_i.get_cat() for name_i in names]
        return np.array(X),y

    def save(self,out_path):
        files.make_dir(out_path)
        for name_i,seq_i in self.items():
            out_i="%s/%s" % (out_path,name_i)
            if( len(self.dims())==3):
                seq_i=[np.concatenate(frame_j.T,axis=0) 
                        for frame_j in seq_i]
            save_frames(out_i,seq_i)

    def seqs_len(self):
        return [len(seq_i) for seq_i in self.values()]

    def min_len(self):
        return min(self.seqs_len())

def read_frame_seqs(in_path,n_split=1):
    frame_seqs=FrameSeqs()
    for path_i in files.top_files(in_path):
        name_i=files.Name(path_i.split('/')[-1]).clean()
        frames=[ read_frame(path_j,n_split) 
                for path_j in files.top_files(path_i)]
        frame_seqs[name_i]=frames
    return frame_seqs

def read_frame(in_path,n_split=1):
    frame_ij=cv2.imread(in_path,cv2.IMREAD_GRAYSCALE)
    if(n_split==1):
        return frame_ij
    return np.array(np.vsplit(frame_ij,n_split)).T
#    return np.array(np.vsplit(frame_ij,n_split)[0]).T

def save_frames(in_path,frames):
    files.make_dir(in_path)
    for i,frame_i in enumerate(frames):
        out_i="%s/%d.png" % (in_path,i)
        cv2.imwrite(out_i, frame_i)

def extract_features(in_path,nn_path,out_path):
    frame_seqs=read_frame_seqs(in_path)
    model=load_model(nn_path)   
    feat_seqs=seqs.Seqs()
    for name_i,seq_i in frame_seqs.items():
        feat_seqs[name_i]=model.predict(np.array(seq_i))
    feat_seqs.save(out_path)