import sys
sys.path.append("..")
import files,tools,action_imgs

def bound_exp(in_path,out_path):
	paths=files.get_paths(out_path,["actions","bounds"])
	action_imgs.mean_action(in_path,paths["actions"],dims=None)
	tools.cut_actions(paths["actions"],paths["bounds"],scale=None,binary=True)

in_path="../../clean/frames"
out_path="../../clean"
bound_exp(in_path,out_path)