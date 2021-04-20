from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

class SimConv(object):
    def __init__(self,name="hidden",activ='relu',l1=0.01):
        self.activ=activ
        self.name=name
        self.l1=l1

    def __call__(self,model,params):
        model.add(Conv2D(64, kernel_size=(5, 5),
                 activation=self.activ))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(32, kernel_size=(5, 5),
                 activation=self.activ))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        reg= None if(self.l1 is None) else regularizers.l1(0.01)
        model.add(Dense(128, activation=self.activ,name=self.name,
            kernel_regularizer=reg,))
        return model