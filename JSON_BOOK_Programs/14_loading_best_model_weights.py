from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)

Dataset = numpy.loadtxt("diabetes.csv", delimiter=",")
X = Dataset[:, 0:8]
Y = Dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

#load weights
model.load_weights("weights-best.hdf5")

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))