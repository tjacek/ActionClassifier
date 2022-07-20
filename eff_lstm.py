import data.paths

def train(in_path):
    path_dict=data.paths.read_paths(in_path)
    print(path_dict)

in_path="../CZU-MHAD/CZU-MHAD/final"
train(in_path)