import numpy as np
import files

class Seqs(dict):
	def __init__(self):
		super(Seqs, self).__init__()

def read_seqs(in_path):
	paths=files.top_files(in_path)
	seq_dict=Seqs()
	for path_i in paths:
		data_i=np.load(path_i)
		name_i=path_i.split('/')[-1]
		name_i=files.clean(name_i)
		seq_dict[name_i]=data_i
	return seq_dict

read_seqs("Data/MSR/common/seqs")	
