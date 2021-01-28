from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

def make_conv(model,params):
#    model = Sequential()
    params['name']="large_dense"
    model=base_conv(model,params)
    model.add(Dense(128, activation='relu',name="hidden"))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=params['n_cats'], activation='softmax'))
#    model.compile(loss=keras.losses.categorical_crossentropy,
#              #optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
#              optimizer=keras.optimizers.Adadelta())
#    model.summary()
    return model

def base_conv(model,params):
    model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu'))#,
#                 input_shape=params['dims']))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu'))#,input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    name=params["name"] if(params['name']) else "hidden"
    model.add(Dense(128, activation='relu',name=name,
    	kernel_regularizer=regularizers.l1(0.01),))
    return model
