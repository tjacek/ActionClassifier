import numpy as np
import re,cv2
import data

def read_imgs(in_path,n_split=4):
    action_dirs=data.top_files(in_path)
    action_dirs.sort(key=data.natural_keys)
    X,y=[],[]
    for action_i in action_dirs:
        person_i= get_person(action_i)
        frame_path=data.top_files(action_i)
        frame_path.sort(key=data.natural_keys)
        for frame_ij_path in frame_path:
            frame_ij=cv2.imread(frame_ij_path,0)
            frame_ij=np.array(np.vsplit(frame_ij,n_split)).T
            X.append(frame_ij)
            y.append( person_i)
    return np.array(X),y

def get_person(action_i):
    name_i=action_i.split("/")[-1]
    name_i=re.sub('[a-z]','',name_i)
    return int(name_i.split('_')[-1])

X,y=read_imgs('mra')
print(X.shape)