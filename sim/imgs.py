from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

def make_conv(model):
#    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu'))#,
#                 input_shape=params['dims']))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu'))#,input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=params['n_cats'], activation='softmax'))
#    model.compile(loss=keras.losses.categorical_crossentropy,
#              #optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
#              optimizer=keras.optimizers.Adadelta())
#    model.summary()
    return model