import numpy as np
import files,data.feats

class Seqs(dict):
	def __init__(self,args=[]):
		super(Seqs, self).__init__(args)

	def seq_len(self):
		return [ len(seq_i) for seq_i in self.values()]

	def equal_seqs(self):
		lengths=self.seq_len()
		return all(l_i==lengths[0] for l_i in lengths)

	def dim(self):
		return list(self.values())[0].shape[1]

	def names(self):
		return sorted(self.keys(),key=files.natural_keys) 
	
	def split(self,selector=None):
		train,test=files.split(self,selector)
		return Seqs(train),Seqs(test)

	def to_dataset(self):
		names=self.names() 
		X=[self[name_i] for name_i in names]
		if(self.equal_seqs()):
			X=np.array(X)
		y=[name_i.get_cat() for name_i in names]#int(name_i.split('_')[0])-1 for name_i in names]
		return X,y

	def to_feats(self,fun):
		feat_dict=data.feats.Feats()
		for name_i,seq_i in self.items():
			feat_dict[name_i]=fun(seq_i)
		return feat_dict

	def save(self,out_path):
		files.make_dir(out_path)
		for name_i,seq_i in self.items():
			out_i="%s/%s" % (out_path,name_i)
			np.save(out_i,seq_i)

def read_seqs(in_path):
	paths=files.top_files(in_path)
	seq_dict=Seqs()
	for path_i in paths:
		data_i=np.load(path_i)
		name_i=path_i.split('/')[-1]
		name_i=files.Name(name_i)#clean(name_i)
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