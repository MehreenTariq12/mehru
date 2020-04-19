import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

numpy.random.seed(7)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

seq_length = 1
dataX = []
dataY = []
for i in range(len(alphabet)-seq_length):
    inp_seq = alphabet[i:i+seq_length]
    out_Seq = alphabet[i+seq_length]
    dataX.append([char_to_int[char] for char in inp_seq])
    dataY.append(char_to_int[out_Seq])
    print(inp_seq, "-->", out_Seq)

X = numpy.reshape(dataX, (len(dataX), seq_length, 1))

X = X/float(len(alphabet))

Y = np_utils.to_categorical(dataY)

batch_size = 1
model = Sequential()
model.add(LSTM(16, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
for i in range(300):
    model.fit(X, Y, epochs = 1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

scores = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
model.reset_states()
print("Model Accuracy: %.2f%%" % (scores[1]*100))
seed = [char_to_int[alphabet[0]]]
for i in range(0, len(alphabet)-1):
    x = numpy.reshape(seed, (1, len(seed), 1))
    x=x/float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()

#random starting
letter = "M"
seed = [char_to_int[letter]]
for i in range(5):
    x = numpy.reshape(seed, (1, len(seed), 1))
    x = x/float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    print(int_to_char[seed[0]], "-->", int_to_char[index])
    seed = [index]
model.reset_states()
# This model has not predicted well with random start
# it needs sequence from A to L before M to predict M correctly
