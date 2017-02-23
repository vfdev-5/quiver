# quiver example

# Python 
import os

# Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD

# Quiver
import sys
sys.path.append('../')
from quiver_engine import new_server as server

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

labels = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ] 

# CNN : Conv3x3 -> Pool -> Dense
model = Sequential()
model.add(Convolution2D(12, 3, 3, activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

server.launch(model=model, inputs=X_train[:5], classes=labels)

