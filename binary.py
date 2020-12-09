import numpy as np
import files,ens,feats,one_shot

def binary_one_shot(in_path,ens_path,n_epochs=100):
    files.make_dir(ens_path)
    binary_sim(in_path,ens_path,n_epochs=n_epochs)
    funcs=[[one_shot.dtw_extract,["in_path","nn","feats"]]]
    dir_names=["feats"]
    ensemble=ens.EnsTransform(funcs,dir_names,"nn")
    arg_dict={'in_path':in_path}
    input_paths=[path_i for path_i in files.top_files("%s/nn"%ens_path)]
    ensemble(input_paths,ens_path, arg_dict)

def binary_sim(in_path,nn_path,n_epochs=5):
	dtw_feats=feats.read_feats(in_path)
	train,test=dtw_feats.split()
	nn_path="%s/nn" % nn_path
	files.make_dir(nn_path)
	for i in range(train.n_cats()):
		nn_i="%s/%d" % (nn_path,i)
		X,y=binary_data(train,i)
		params={'input_shape':(train.dim(),)}
		siamese_net,extractor=one_shot.build_siamese(params,one_shot.DtwModel())
		siamese_net.fit(X,y,epochs=n_epochs,batch_size=64)
		extractor.save(nn_i)

def binary_data(train,cat):
	X,y=train.to_dataset()
	size=len(y)
	new_X,new_y=[],[]
	for i in range(size):
		if(y[i]==cat):
			for j in range(size):
				new_X.append( (X[i],X[j]))
				new_y.append( (y[j]==cat))
	new_X=np.array(new_X)
	new_X=[new_X[:,0],new_X[:,1]]
	return new_X,new_y

#def ens_extract(frame_path,nn_path,out_path):
#	funcs=[[imgs.extract_features,["frames","models","seqs"]]]
#	dir_names=["seqs"]
#	ens=EnsTransform(funcs,dir_names,"models")
#	arg_dict={"frames":frame_path}
#	input_paths=files.top_files(nn_path)
#	ens(input_paths,out_path, arg_dict)

binary_one_shot(["dtw/corl/feats","dtw/maxz/feats"],"ens",n_epochs=100)