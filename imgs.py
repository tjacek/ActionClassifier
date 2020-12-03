import numpy as np
import cv2
from keras.models import load_model
import files,seqs

class FrameSeqs(dict):
    def __init__(self, args=[]):
        super(FrameSeqs, self).__init__(args)

    def dim(self):
        return list(self.values())[0][0].shape

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return FrameSeqs(train),FrameSeqs(test)

    def save(self,out_path):
        files.make_dir(out_path)
        for name_i,seq_i in self.items():
            out_i="%s/%s" % (out_path,name_i)
            seq_i=[np.concatenate(frame_j.T,axis=0) 
                    for frame_j in seq_i]
            save_frames(out_i,seq_i)

def read_frame_seqs(in_path):
    frame_seqs=FrameSeqs()
    for path_i in files.top_files(in_path):
        name_i=files.clean(path_i.split('/')[-1])
        frames=[ read_frame(path_j,n_split=3) 
                for path_j in files.top_files(path_i)]
        frame_seqs[name_i]=frames
    return frame_seqs

def read_frame(in_path,n_split=3):
    frame_ij=cv2.imread(in_path,cv2.IMREAD_GRAYSCALE)
    return np.array(np.vsplit(frame_ij,n_split)).T

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
#def extract_frame_feats(in_path,nn_path,out_path):
#    model=load_model(nn_path)
#    action_dirs=data.top_files(in_path)
#    action_dirs.sort(key=data.natural_keys)
#    X,y=[],[]
#    data.make_dir(out_path)
#    for action_i in action_dirs:
#        frame_path=data.top_files(action_i)
#        frame_path.sort(key=data.natural_keys)
#        def frame_helper(path_ij):
#            frame_ij=read_frame(path_ij)
#            frame_ij=np.expand_dims(frame_ij,0)
#            return model.predict(frame_ij)
#        pred_i=np.array([frame_helper(path_ij) for path_ij in frame_path])
#        out_i=out_path+'/'+action_i.split('/')[-1]
#        pred_i=np.squeeze(pred_i, axis=1)
#        np.savetxt(out_i, pred_i, delimiter=',' )