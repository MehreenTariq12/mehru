import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

numpy.random.seed(7)

#raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#charcter to integer and reverse maping
char_to_int = dict((c,i) for (i, c) in enumerate(alphabet))
int_to_char = dict((i,c) for (i, c) in enumerate(alphabet))

#prepare dataset in form of input output pairs
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    input_Seq = alphabet[i:i + seq_length]
    output_Seq = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in input_Seq])
    dataY.append(char_to_int[output_Seq])
    #print(input_Seq, "-->", output_Seq)

#reshape for LSTM
X = pad_sequences(dataX, maxlen=seq_length, dtype="float32")

X = numpy.reshape(X, (len(dataX), seq_length, 1))

X = X/float(len(alphabet))

Y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X, Y, epochs=5000, batch_size=len(dataX), verbose=2, shuffle = False)

scores = model.evaluate(X, Y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

for pattern in dataX:
    x= numpy.reshape(pattern, (1, len(pattern), 1))
    x = x/float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "-->", result)
print("Test random patterns")
for i in range(0,20):
    pattern_index = numpy.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "-->", result)

