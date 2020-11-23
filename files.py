import os,re

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

def make_path(path):
    paths=path.split("/")
    for i in range(len(paths)):
        path_i="/".join(paths[:i+1])
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

def recursive_transform(in_path,out_path,name,fun):
    def helper(root):
        out_i=replace_path(root,out_path)            
        make_path(out_i)
        fun(root,out_i)
    for (root,dirs,files) in os.walk(in_path, topdown=True): 
        if(root.split('/')[-1]==name):
            if(dirs):
                for path_j in dirs:
                    in_i="%s/%s" % (root,path_j)
                    helper(in_i)
            else:
                helper(root)

def get_paths(dir_path,sufixes):
    return {sufix_i:"%s/%s"%(dir_path,sufix_i) for sufix_i in sufixes}

def ens_template(in_path,out_path,fun):
    make_dir(out_path)
    for in_i in top_files(in_path):
        out_i="%s/%s" % (out_path,in_i.split('/')[-1])
        fun(in_i,out_i)