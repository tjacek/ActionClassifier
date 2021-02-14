import sys
sys.path.append("..")
import files,tools,action_imgs
import rect

def gen_frames(old_frames,dataset_path,dir_path):
	files.make_dir(dir_path)
	new_frames="%s/frames" % dir_path
	rect.dataset_cut(dataset_path,old_frames,new_frames)
	show_frames(new_frames,dir_path)

def show_frames(frame_path,dir_path):
	paths=files.get_paths(dir_path,["actions","bounds"])
	action_imgs.mean_action(frame_path,paths["actions"],dims=None)
	tools.cut_actions(paths["actions"],paths["bounds"],scale=None,binary=True)

frame_path="../../clean/exp2/frames"
dataset_path="../../clean/exp2/dataset"
exp_path="../../clean/exp3"

#bound_exp(frame_path,dir_path)
#rect.make_dataset(action_path,dataset_path)
gen_frames(frame_path,dataset_path,exp_path)