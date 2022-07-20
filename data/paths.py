from . import DataDict

class PathsDict(DataDict):
    def __init__(self, args=[]):
        super(DataDict, self).__init__(args)

    def read(self,name_i):
        return imgs.read_frames(self[name_i])


def read_paths(in_path):
    path_dict=PathsDict()
    