import numpy as np
import files

class Seqs(dict):
	def __init__(self,args=[]):
		super(Seqs, self).__init__(args)

	def split(self,selector=None):
		if(not selector):
			selector=person_selector
		train,test=Seqs(),Seqs()
		for name_i in self.keys():
			if(selector(name_i)):
				train[name_i]=self[name_i]
			else:
				test[name_i]=self[name_i]
		return train,test

	def to_dataset(self):
		names=sorted(self.keys(),key=files.natural_keys) 
		X=[self[name_i] for name_i in names]
		y=[ int(name_i.split('_')[0])-1 for name_i in names]
		return X,y

	def save(self,out_path):
		files.make_dir(out_path)
		for name_i,seq_i in self.items():
			out_i="%s/%s" % (out_path,name_i)
			np.save(out_i,seq_i)

def person_selector(name_i):
	person_i=int(name_i.split('_')[1])
	return person_i%2==1

def read_seqs(in_path):
	paths=files.top_files(in_path)
	seq_dict=Seqs()
	for path_i in paths:
		data_i=np.load(path_i)
		name_i=path_i.split('/')[-1]
		name_i=files.clean(name_i)
		seq_dict[name_i]=data_i
	return seq_dict

def rec_split(in_path,out_path):
	def helper(in_i,out_i):
		print(in_i)
		print(out_i)
		seqs_i=read_seqs(in_i)
		train,test=seqs_i.split()
		train.save("%s/train" % out_i)
		test.save("%s/test" % out_i)
	files.recursive_transform(in_path,out_path,"seqs",helper)