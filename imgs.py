import numpy as np
import re,cv2
import data

def train_model(in_path,out_path,out_path,n_epochs=10):
    train_X,y,test_X,test_y,params=prepare(in_path)
    model=make_model(params)
    model.summary()
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    model.binary(out_path)

def read_imgs(in_path,n_split=4):
    action_dirs=data.top_files(in_path)
    action_dirs.sort(key=data.natural_keys)
    X,y=[],[]
    for action_i in action_dirs:
        person_i= get_person(action_i)
        if((person_i%2)==1):
            frame_path=data.top_files(action_i)
            frame_path.sort(key=data.natural_keys)
            for frame_ij_path in frame_path:
                frame_ij=cv2.imread(frame_ij_path,0)
                frame_ij=np.array(np.vsplit(frame_ij,n_split)).T
                X.append(frame_ij)
                y.append( person_i)
    return np.array(X),y

def get_person(action_i):
    name_i=action_i.split("/")[-1]
    name_i=re.sub('[a-z]','',name_i)
    return int(name_i.split('_')[1])

def make_conv(params):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu')
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',name="hidden"))
    model.add(Dropout(0.5))
    model.add(Dense(units=params['n_cats'], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    return model


train_model("mra","persons",n_epochs=10)