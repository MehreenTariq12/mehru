from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt

seed = 7
numpy.random.seed(seed)

Dataset = numpy.loadtxt("diabetes.csv", delimiter=",")
X = Dataset[:, 0:8]
Y = Dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('model accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('model loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()