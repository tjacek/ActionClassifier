import files

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

def ens_template(in_path,out_path,fun):
    files.make_dir(out_path)
    for in_i in files.top_files(in_path):
        out_i="%s/%s" % (out_path,in_i.split('/')[-1])
        fun(in_i,out_i)