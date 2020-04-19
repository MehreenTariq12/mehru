from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import numpy
import os

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("diabetes.csv", delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,Y,epochs=150, batch_size=10, verbose=0)

scores = model.evaluate(X,Y,verbose=0)

print("%s %.2f%%" % (model.metrics_names[1], scores[1]*100))

model_yaml = model.to_yaml()
with open("model.yaml","w") as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights("model.h5")
print("yaml model saved to disk")

yaml_file = open("model.yaml", "r")
yaml_file_model = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(yaml_file_model)
loaded_model.load_weights("model.h5")
print("model loaded from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


