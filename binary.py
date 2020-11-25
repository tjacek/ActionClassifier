import files,spline,convnet

class SeqEnsemble(object):
	def __init__(self,funcs,dir_names):
		self.funcs=funcs
		self.dir_names=dir_names

	def __call__(self,input_paths,out_path, arg_dict):
		dirs=files.get_paths(out_path,self.dir_names)
		for dir_i in dirs.values():
			files.make_dir(dir_i)
		for path_i in input_paths:
			name_i=path_i.split('/')[-1]
			args_i={ key_i:"%s/%s" % (path_i,name_i) 
				for key_i,path_i in dirs.items()}
			args_i={**args_i,**arg_dict}
			args_i["seqs"]=path_i
			print(path_i)
			for fun,arg_names in self.funcs:
				fun_args=[args_i[name_k]  
							for name_k in arg_names]
				fun(*fun_args)

def binary_exp(in_path,n_epochs=1000):
	binary_path="%s/binary" % in_path
	files.make_dir(binary_path)
	input_paths=files.top_files("%s/seqs" % in_path)
	ensemble1D(input_paths,binary_path)
#	binary(input_paths,binary_path)

def ensemble1D(input_paths,out_path,n_epochs=1000,size=96):
	funcs=[ [spline.upsample,["seqs","spline","size"]],
			[convnet.train_nn,["spline","nn","n_epochs"]],
			[convnet.extract,["spline","nn","feats"]]]
	dir_names=["spline","nn","feats"]
	ens=SeqEnsemble(funcs,dir_names)
	arg_dict={'size':96,'n_epochs':n_epochs}
	ens(input_paths,out_path, arg_dict)

binary_exp("Data/3DHOI")