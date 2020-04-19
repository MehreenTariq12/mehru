from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
seed = 7
numpy.random.seed(seed)
#load data
dataset = numpy.loadtxt("diabetes.csv", delimiter=",")
#split in attributes and class
X = dataset[:, 0:8]
Y = dataset[:, 8]
#split train 67% and validation 33% data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
#create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu' ))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
#Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fit model
#model.fit(X, Y, nb_epoch=150, batch_size=10, validation_split=0.33)
model.fit(X_train, Y_train, nb_epoch=150, batch_size=10, validation_data=(X_test, Y_test))
#evaluate model
scores = model.evaluate(X,Y)
print("%s:   %.2f%%" % (model.metrics_names[1], scores[1]*100))