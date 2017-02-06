# quiver example

# Python 
import os

tmp_path='examples/tmp'
if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)

# Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD

# Quiver
from quiver_engine import server

# CNN : Conv3x3 -> Pool -> Dense
model = Sequential()
model.add(Convolution2D(12, 3, 3, activation='relu', input_shape=(120, 120, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

server.launch(model, classes=['cat', 'dog'], temp_folder=tmp_path, input_folder='examples/data')

