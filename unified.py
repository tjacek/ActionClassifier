import numpy as np
import action_imgs

def agum_action(in_path,out_path):
    actions=action_imgs.read_actions(in_path)
    train,test=actions.split()
    agum=train.transform(flip_agum,copy=True)
    for name_i,img_i in agum.items():
        agum_i="%s_1" % name_i
        actions[agum_i]=img_i
    actions.save(out_path)

def flip_agum(img_i):
    return np.flip(img_i,1)

agum_action("../3DHOI/mean","../3DHOI_agum/mean")