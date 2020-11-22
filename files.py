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

def make_path(path):
    paths=path.split("/")
    for i in range(len(paths)):
        path_i="/".join(paths[:i+1])
        print(path_i)
        make_dir(path_i)

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def replace_path(in_path,out_path):
    paths=in_path.split('/')[1:]
    paths=[out_path]+paths
    return "/".join(paths)

def clean(name_i):
    raw=name_i.split('_')
    name_i=re.sub(r'\D0','',name_i.strip())
    return "_".join(re.findall(r'\d+',name_i))

def recursive_transform(in_path,out_path,name):
    for (root,dirs,files) in os.walk(in_path, topdown=True): 
        if(root.split('/')[-1]==name):
            out_i=replace_path(root,out_path)            
            make_path(out_i)
            print("--------------")    
            print(root)
            print(dirs)
            print(out_i)