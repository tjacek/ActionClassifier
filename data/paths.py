import os
import files
from . import DataDict,imgs

class PathsDict(DataDict):
    def __init__(self, args=[],reader=None):
        super(DataDict, self).__init__(args)
        if(reader is None):
            reader= imgs.ReadFrames()
        self.reader=reader

    def read(self,name_i):
        frames=[self.reader(path_i) 
            for path_i in self[name_i]]
        return np.array(frames)

def read_paths(in_path):
    path_dict=PathsDict()
    for name_i in os.listdir(in_path):
        path_i=f"{in_path}/{name_i}"
        path_dict[name_i]=files.top_files(path_i)  
    return path_dict