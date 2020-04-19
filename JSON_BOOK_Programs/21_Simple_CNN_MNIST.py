import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K

seed = 7
numpy.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train/255
X_test = X_test/255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_train.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def larger_CNN():
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape=(28, 28, 1), kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3,3), kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(60, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(10,  activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = larger_CNN()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
