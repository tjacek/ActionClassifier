import files,spline,convnet,imgs

class EnsTransform(object):
	def __init__(self,funcs,dir_names,input_dir="seqs"):
		self.funcs=funcs
		self.dir_names=dir_names
		self.input_dir=input_dir

	def __call__(self,input_paths,out_path, arg_dict):
		dirs=files.get_paths(out_path,self.dir_names)
		for dir_i in dirs.values():
			files.make_dir(dir_i)
		for path_i in input_paths:
			name_i=path_i.split('/')[-1]
			args_i={ key_i:"%s/%s" % (path_i,name_i) 
				for key_i,path_i in dirs.items()}
			args_i={**args_i,**arg_dict}
			args_i[self.input_dir]=path_i
			print(path_i)
			for fun,arg_names in self.funcs:
				fun_args=[args_i[name_k]  
							for name_k in arg_names]
				fun(*fun_args)

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