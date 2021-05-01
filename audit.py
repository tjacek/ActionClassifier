from keras.models import load_model
import data.imgs,data.seqs,data.actions,files,sim

def ts_imgs(in_path,out_path):
    seq_dict=data.seqs.read_seqs(in_path)
    ts_imgs=data.actions.ActionImgs()
    for name_i,seq_i in seq_dict.items():
        ts_imgs[name_i]=seq_i*100
    ts_imgs.save(out_path)

def seq_stats(in_path):
    frames=data.imgs.read_frame_seqs(in_path,n_split=1)
    seqs_len=frames.seqs_len()
    stats=(sum(seqs_len),min(seqs_len),max(seqs_len))
    print("n_frames:%s\nmin_len:%s\nmax_len:%s\n" % stats)
    stats=(len(seqs_len),frames.n_persons())
    print("n_seqs:%s\nn_persons:%s\n" % stats)

def mean_action_size(in_path):
    frames=data.imgs.read_frame_seqs(in_path,n_split=1)
    cats={}
    for name_i,seq_i in frames.items():
        cat_i=name_i.get_cat()
        if( name_i.get_person()%2==1):
            if(not cat_i in cats ):
                cats[cat_i]=0
            cats[cat_i]+=len(seq_i)
    print(cats)
    values=list(cats.values())
    print(np.median(values))
    print(sum(values))

def dataset_size(in_path):
    seq_dict=data.seqs.read_seqs(in_path)
    train,test=seq_dict.split()
    print("n_feats %d" % train.dim())
    print("train %d" % len(train))
    print("test %d" %len(test))

def n_pairs(in_path):
    seq_dict=data.seqs.read_seqs(in_path)
    train,test=seq_dict.split()
    pairs=sim.all_pairs(train.names())
    print("n_pairs:%d" % len(pairs))
    cats=[ sim.all_cat(name_i,name_j) 
            for name_i,name_j in pairs]
    print("Positive %d" % sum(cats))
    print("Negative %d" % (len(cats)-sum(cats)))

def check_model(nn_path):
    model=load_model(nn_path)
    model.summary()

if __name__=="__main__":
    in_path="../ICSS/3DHOI/ens/nn/0"
    check_model(in_path)