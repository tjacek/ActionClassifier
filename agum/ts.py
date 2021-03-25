import sys
sys.path.append("..")
import data.seqs,files,spline,ts_cnn

class TSAgum(object):
	def __init__(self, agum_fun):
		self.agum_fun=agum_fun
		
	def __call__(self,in_path,out_path):
		seq_dict=data.seqs.read_seqs(in_path)
		train,test=seq_dict.split()
		agum_seqs=data.seqs.Seqs()
		agum_fun=[]
		for name_i,seq_i in train.items():
			for k,fun_k in enumerate(self.agum_fun):
				name_k="%s_%d" % (name_i,k)
				agum_seqs[name_k]=fun_k(seq_i)
				agum_seqs[name_i]=seq_i
		full_seqs=data.seqs.Seqs({**agum_seqs, **test})
		full_seqs.save(out_path)

def ts_scale(seq_i):
	return 2*seq_i

def agum_exp(input_path,out_path,n_epochs=1000):
	files.make_dir(out_path)
	paths=files.get_paths(out_path,["spline","agum","nn","feats"])
	spline.upsample(input_path,paths["spline"],size=64)
	agum=TSAgum([ts_scale])
	agum(paths["spline"],paths["agum"])
	train,extract=ts_cnn.get_train()
	train(paths["agum"],paths["nn"],n_epochs)
	extract(paths["agum"],paths["nn"],paths["feats"])

seq_path="../../dtw_paper/MSR/binary/seqs/nn0"
#ensemble_exp(seq_path,"ens_test")
agum_exp(seq_path,"agum",n_epochs=1000)