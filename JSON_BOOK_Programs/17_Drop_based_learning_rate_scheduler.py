import math
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

def step_decay(epoch):
    inital_lr = 0.1
    drop_rate = 0.5
    epoch_drop = 10.0
    lr = inital_lr * math.pow(drop_rate, math.floor((1+epoch)/epoch_drop))
    return lr

seed =7
numpy.random.seed(seed)

dataframe = pandas.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:34].astype(float)
Y = dataset[:, 34]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
model.add(Dense(1,  activation='sigmoid', kernel_initializer='normal'))

sgd = SGD(learning_rate=0.0, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model.fit(X, encoded_Y, validation_split=0.33, epochs=50, batch_size=28, callbacks=callbacks_list, verbose=2)