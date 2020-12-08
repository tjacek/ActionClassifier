import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import files

class Feats(dict):
	def __init__(self, arg=[]):
		super(Feats, self).__init__(arg)
	
	def names(self):
		return sorted(self.keys(),key=files.natural_keys) 
	
	def split(self,selector=None):
		train,test=files.split(self,selector)
		return Feats(train),Feats(test)

	def to_dataset(self):
		names=self.names() 
		X=np.array([self[name_i] for name_i in names])
		y=[ int(name_i.split('_')[0])-1 for name_i in names]
		return X,y

	def norm(self):
		X,y=self.to_dataset()
		mean=np.mean(X,axis=0)
		std=np.std(X,axis=0)
		for name_i,feat_i in self.items():
			self[name_i]=(feat_i-mean)/std

def train_model(feats):
	feats.norm()
	train,test=feats.split()
	model=LogisticRegression(solver='liblinear')
	X_train,y_train=train.to_dataset()
	model.fit(X_train,y_train)
	X_test,y_test=test.to_dataset()
	y_pred=model.predict(X_test)
	print(classification_report(y_test, y_pred,digits=4))

def read_feats(in_path):
    lines=open(in_path,'r').readlines()
    feat_dict=Feats()
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.clean(info_i)
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    return feat_dict

d=read_feats("dtw/maxz/feats")
train_model(d)