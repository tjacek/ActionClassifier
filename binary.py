import files,spline,convnet

def binary_exp(in_path,n_epochs=1000):
	binary_path="%s/binary" % in_path
	files.make_dir(binary_path)
	input_paths=files.top_files("%s/seqs" % in_path)
	binary(input_paths,binary_path)

def binary(input_paths,out_path,n_epochs=1000):
	dirs=files.get_paths(out_path,["spline","nn","feats"])
	for dir_i in dirs.values():
		files.make_dir(dir_i)
	for path_i in input_paths:
		name_i=path_i.split('/')[-1]
		out_i={ key_i:"%s/%s" % (path_i,name_i) 
				for key_i,path_i in dirs.items()}
		spline.upsample(path_i,out_i["spline"],size=96)
		convnet.train_nn(out_i["spline"],out_i["nn"],n_epochs)
		convnet.extract(out_i["spline"],out_i["nn"],out_i["feats"])

binary_exp("Data/MSR")