import numpy as np,re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import files
from . import DataDict

class Feats(DataDict):
	def __init__(self, arg=[]):
		super(Feats, self).__init__(arg)

	def __add__(self,feat_i):
		names=common_names(self.keys(),feat_i.keys())
		new_feats=Feats()
		for name_i in names:
			x_i=np.concatenate([self[name_i],feat_i[name_i]],axis=0)
			new_feats[name_i]=x_i
		return new_feats

	def dim(self):
		return list(self.values())[0].shape[0]

	def to_dataset(self):
		names=self.names()
		X=np.array([self[name_i] for name_i in names])
		return X,self.get_cats()
	
	def transform(self,extractor):
		feat_dict={	name_i: extractor(feat_i)
				for name_i,feat_i in self.items()}
		return Feats(feat_dict)

	def norm(self):
		X,y=self.to_dataset()
		mean=np.mean(X,axis=0)
		std=np.std(X,axis=0)
		std[np.isnan(std)]=1
		std[std==0]=1
		for name_i,feat_i in self.items():
			self[name_i]=(feat_i-mean)/std

	def save(self,out_path):
		lines=[]
		for name_i,feat_i in self.items():
			txt_i=np.array2string(feat_i,separator=",")
			txt_i=txt_i.replace("\n","")
			lines.append("%s#%s" % (txt_i,name_i))
		feat_txt='\n'.join(lines)
		feat_txt=feat_txt.replace('[','').replace(']','')
		feat_txt = feat_txt.replace(' ','')
		file_str = open(out_path,'w')
		file_str.write(feat_txt)
		file_str.close()

def train_model(feats):
	if(type(feats)==str):
		feats=read_feats(feats)
	feats.norm()
	train,test=feats.split()
	model=LogisticRegression(solver='liblinear')
	X_train,y_train=train.to_dataset()
	model.fit(X_train,y_train)
	X_test,y_test=test.to_dataset()
	y_pred=model.predict(X_test)
	print(classification_report(y_test, y_pred,digits=4))
	print(accuracy_score(y_test,y_pred))

def read_feats(in_path):
    if(type(in_path)==list):
        all_feats=[read_feats(path_i) for path_i in in_path]
        return concat_feats(all_feats)
    lines=open(in_path,'r').readlines()
    feat_dict=Feats()
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.Name(info_i).clean()#files.clean(info_i)
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    return feat_dict

def concat_feats(all_feats):
	first=all_feats[0]
	for feat_i in all_feats[1:]:
		first+=feat_i
	return first

def common_names(names1,names2):
	return list(set(names1).intersection(set(names2)))

def unified_exp(in_path):
	all_feats=read_feats(files.top_files(in_path))
	train_model(all_feats)

if __name__ == "__main__":
#	d=read_feats(["dtw/maxz/feats","dtw/corl/feats","dtw/skew/feats"])
#	d=read_feats(['../agum/max_z/dtw','../agum/corl/dtw','../agum/skew/dtw'])
#	d=read_feats(['../agum/max_z/person','../agum/corl/person','../agum/skew/person'])
	d=read_feats(['../agum/max_z/cat','../agum/corl/cat','../agum/skew/cat'])
	train_model(d)#"sim_feats")
#	unified_exp("../agum/ens/feats")