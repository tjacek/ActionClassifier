import sys
sys.path.append("..")
import data.actions,gui

class ActionState(object):
	def __init__(self, actions):
		self.actions=actions

def make_dataset(in_path):
	action_imgs=data.actions.read_actions(in_path)
	state=ActionState(action_imgs)
	gui.gui_exp(state)
#	print(len(action_imgs))

in_path="../../clean/mean"
make_dataset(in_path)