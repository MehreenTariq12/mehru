import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD

seed =7
numpy.random.seed(seed)

dataframe = pandas.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:34]
Y = dataset[:, 34]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
model.add(Dense(1,  activation='sigmoid', kernel_initializer='normal'))
learning_rate = 0.1
epochs = 50
decay = learning_rate/epochs
momentum = 0.8
sgd = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X, encoded_Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2)