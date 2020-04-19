import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers import Conv1D
from keras.layers import MaxPool1D

seed = 7
numpy.random.seed(seed)

top_words = 5000
max_words = 500

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(nb_words=top_words)

X_train= sequence.pad_sequences(X_train, maxlen=max_words)
X_test= sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
print(model.summary())

model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=2, batch_size=128, verbose=1)
scores = model.evaluate(X_test,Y_test,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))