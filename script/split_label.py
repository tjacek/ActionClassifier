import sys
sys.path.append("..")
import data.imgs,files

def split_each_label(in_path,out_path,train_size=0.8):
    frames=data.imgs.read_frame_seqs(in_path)
    new_frames=data.imgs.FrameSeqs()
    for name_i,seq_i in frames.items():
        split_size= int(len(seq_i) *(1.0-train_size))
        n_split=int(len(seq_i) /split_size) 
        for j in range(n_split-1):
            seq_j=seq_i[j*split_size:(j+1)*split_size]
            name_j=files.Name("%d_1_%d" % (int(name_i),j))
            new_frames[name_j]=seq_j
            print(name_j)
        seq_j=seq_i[(n_split-1)*split_size: ]
        name_j=files.Name("%d_2_%d" % (int(name_i),0))
        new_frames[name_j]=seq_j
    new_frames.scale()
    new_frames.save(out_path)

box_path="../../forth/box"
out_path="../../forth/splited"
split_each_label(box_path,out_path,train_size=0.8)