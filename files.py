import os,re

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def clean(self):
        digits=[ str(int(digit_i)) 
                for digit_i in re.findall(r'\d+',self)]
        return Name("_".join(digits))

    def get_cat(self):
        return int(self.split('_')[0])-1

    def get_person(self):
        return int(self.split('_')[1])

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def bottom_files(path,full_paths=True):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            for filename_i in filenames:
                path_i= root+'/'+filename_i if(full_paths) else filename_i
                all_paths.append(path_i)
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def split(dict,selector=None):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in dict.keys():
        if(selector(name_i)):
            train.append((name_i,dict[name_i]))
        else:
            test.append((name_i,dict[name_i]))
    return train,test

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1

#def make_path(path):
#    paths=path.split("/")
#    for i in range(len(paths)):
#        path_i="/".join(paths[:i+1])
#        make_dir(path_i)

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

#def replace_path(in_path,out_path):
#    paths=in_path.split('/')[1:]
#    paths=[out_path]+paths
#    return "/".join(paths)

def get_paths(dir_path,sufixes):
    return {sufix_i:"%s/%s"%(dir_path,sufix_i) for sufix_i in sufixes}

def prepare_dirs(in_path,out_dir,sub_dirs):
    path,name=os.path.split(in_path)
    basic_path="%s/%s" % (path,out_dir)
    make_dir(basic_path)
    paths=get_paths(basic_path,sub_dirs)
    paths[name]=in_path
    return paths