from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

class SimConv(object):
    def __init__(self,name="hidden",activ='relu'):
        self.activ=activ
        self.name=name

    def __call__(self,model,params):
        model.add(Conv2D(64, kernel_size=(5, 5),
                 activation=self.activ))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(32, kernel_size=(5, 5),
                 activation=self.activ))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
#        name=params["name"] if(params['name']) else "hidden"
        model.add(Dense(128, activation=self.activ,name=self.name,
            kernel_regularizer=regularizers.l1(0.01),))
        return model


#def make_conv(model,params):
#    params['name']="large_dense"
#    model=base_conv(model,params)
#    model.add(Dense(128, activation='relu',name="hidden"))
#    return model

#def base_conv(model,params):
#    model.add(Conv2D(64, kernel_size=(5, 5),
#                 activation='relu'))
#    model.add(MaxPooling2D(pool_size=(4, 4)))
#    model.add(Conv2D(32, kernel_size=(5, 5),
#                 activation='relu'))
#    model.add(MaxPooling2D(pool_size=(4, 4)))
#    model.add(Flatten())
#    name=params["name"] if(params['name']) else "hidden"
#    model.add(Dense(128, activation='relu',name=name,
#    	kernel_regularizer=regularizers.l1(0.01),))
#    return model
