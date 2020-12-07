import files,spline,convnet,imgs

def binary_exp(in_path,n_epochs=1000,dir_name="narrow"):
	binary_path="%s/%s" % (in_path,dir_name)
	files.make_dir(binary_path)
	input_paths=files.top_files("%s/seqs" % in_path)
	train,extract=convnet.get_train("narrow")
	ensemble1D(input_paths,binary_path,train,extract)
#	binary(input_paths,binary_path)

def ensemble1D(input_paths,out_path,train,extract,n_epochs=1000,size=64):
	funcs=[ [spline.upsample,["seqs","spline","size"]],
			[train,["spline","nn","n_epochs"]],
			[extract,["spline","nn","feats"]]]
	dir_names=["spline","nn","feats"]
	ens=EnsTransform(funcs,dir_names)
	arg_dict={'size':size,'n_epochs':n_epochs}
	ens(input_paths,out_path, arg_dict)

def ens_extract(frame_path,nn_path,out_path):
	funcs=[[imgs.extract_features,["frames","models","seqs"]]]
	dir_names=["seqs"]
	ens=EnsTransform(funcs,dir_names,"models")
	arg_dict={"frames":frame_path}
	input_paths=files.top_files(nn_path)
	ens(input_paths,out_path, arg_dict)

#binary_exp("Data/MSR")
ens_extract("short/ens/frames","short/models","short/ens")