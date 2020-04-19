import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

#load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#flatten training and test data 60,000*28*28 --> 60,000*784
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#normalizing data
X_train = X_train/255
X_test = X_test/255

#binarizing labels
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X_train, Y_train, epochs=10, batch_size=200, verbose=2, validation_data=(X_test, Y_test))
scores = model.evaluate(X_test,Y_test, verbose=0)
print("Baseline Error %.2f%%" % (100-scores[1]*100))