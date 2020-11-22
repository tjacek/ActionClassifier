import os,re

def top_files(path):
    return [ path+'/'+file_i for file_i in os.listdir(path)]
    	
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

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def clean(name_i):
    raw=name_i.split('_')
#    ending= raw[-1] if(postfix and len(raw)>3) else ''
    name_i=re.sub(r'\D0','',name_i.strip())
    return "_".join(re.findall(r'\d+',name_i))