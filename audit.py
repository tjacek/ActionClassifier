import data.imgs

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

if __name__=="__main__":
#    check_model("../action/ens/nn/0")
    mean_action_size('../../common/frames/full')