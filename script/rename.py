import shutil,os

def rename_dicts(in_path,out_path):
    if(not os.path.isdir(out_path)):
        os.mkdir(out_path)
    for name_i in os.listdir(in_path):
        in_i="%s/%s" % (in_path,name_i)
        out_i="%s/%s" % (out_path,add_cat(name_i))
        print(out_i)
        shutil.move(in_i,out_i)

def add_cat(name_i):
    raw=name_i.split("_")
    cat=str(int(raw[0])+1)
    raw=[cat]+raw[1:]
    return "_".join(raw)

in_path="../../forth/box"
out_path="../../forth/box2"
rename_dicts(in_path,out_path)