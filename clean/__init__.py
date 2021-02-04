import sys
sys.path.append("..")
import data.actions

def make_dataset(in_path):
	action_imgs=data.actions.read_actions(in_path)
	print(len(action_imgs))

in_path="../../clean/mean"
make_dataset(in_path)