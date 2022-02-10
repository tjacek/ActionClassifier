import files

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(DataDict, self).__setitem__(key, value)
    
    def n_cats(self):
        return self.names().n_cats()

    def names(self):
        keys=sorted(self.keys(),key=files.natural_keys) 
        return NameList(keys)

    def split(self,selector=None,names_only=False):
        train,test=split(self,selector)
        if(names_only):
            return train,test
        return self.__class__(train),self.__class__(test)

class NameList(list):
    def __new__(cls, name_list=None):
        if(name_list is None):
            name_list=[]
        return list.__new__(cls,name_list)

    def n_cats(self):
        return len(self.unique_cats())

    def unique_cats(self):
        return set(self.get_cats())

    def get_cats(self):
        return [name_i.get_cat() for name_i in self]     

    def binarize(self,j):
        return [ int(cat_i==0) for cat_i in self.get_cats()]

    def by_cat(self):
        cat_dict={cat_j:NameList() 
                for cat_j in self.unique_cats()}
        for name_i in self:
            cat_dict[name_i.get_cat()].append(name_i)
        return cat_dict

    def cats_stats(self):
        stats_dict={ cat_i:0 for cat_i in self.unique_cats()}
        for cat_i in self.get_cats():
            stats_dict[cat_i]+=1
        return stats_dict

    def subset(self,indexes):
        return NameList([self[i] for i in indexes])



def split(data_dict,selector=None,pairs=True):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in data_dict.keys():
        pair_i=(name_i,data_dict[name_i]) if(pairs) else name_i
        if(selector(name_i)):
            train.append(pair_i)
        else:
            test.append(pair_i)
    return train,test

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1