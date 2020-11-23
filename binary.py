#import numpy as np
#from keras.models import load_model
#import data,lstm
import numpy as np
import seqs,files

def unified_exp(in_path):
    all_seqs=[seqs.read_seqs(path_i) 
        for path_i in files.top_files(in_path)]
    names=all_seqs[0].names()
    unified=seqs.Seqs()
    for name_i in names:
        seq_i=[ dict_j[name_i]  for dict_j in all_seqs]
        unified[name_i]=np.concatenate(seq_i,axis=1)
    print(unified.dim())

unified_exp("Data/MSR/spline")
#def extract_binary(in_path,nn_path,out_path):
#    binary_paths=data.bottom_files(nn_path)
#    data.make_dir(out_path)
#    data_dict=data.get_dataset(in_path,splited=False)
#    for path_i in binary_paths:
#    	print(path_i)
#        model_i=load_model(path_i)
#        extr_i=lstm.make_extractor(model_i)
#        feat_dict_i=lstm.lstm_extraction(data_dict,extr_i)
#        feat_path_i=out_path+'/'+path_i.split('/')[-1]
#        data.save_feats(feat_dict_i,feat_path_i)

#def train_binary_models(in_path,out_path,n_epochs=10):
#    data.make_dir(out_path)
#    train_X,train_y,test_X,test_y,params=lstm.prepare_data(in_path)
#    all_cats=params['n_cats']
#    params['n_cats']=2    
#    for cat_i in range(all_cats):
#        print("model category %d" % cat_i)
#        binary_y=binarize(train_y,cat_i)
#        model_i=lstm.make_conv_lstm(params)
#        model_i.fit(train_X,binary_y,epochs=n_epochs,batch_size=32)
#        out_i=out_path+'/nn'+str(cat_i)
#        model_i.save(out_i)