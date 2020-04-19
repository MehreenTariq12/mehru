import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

numpy.random.seed(7)

#raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#charcter to integer and reverse maping
char_to_int = dict((c,i) for (i, c) in enumerate(alphabet))
int_to_char = dict((i,c) for (i, c) in enumerate(alphabet))

#prepare dataset in form of input output pairs
#seq_length = 1
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    input_Seq = alphabet[i:i + seq_length]
    output_Seq = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in input_Seq])
    dataY.append(char_to_int[output_Seq])
    #print(input_Seq, "-->", output_Seq)

#reshape for LSTM
#X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
X = numpy.reshape(dataX, (len(dataX), 1, seq_length))

#normalize
X = X / float(len(alphabet))

#one hot encoding of labels
Y = np_utils.to_categorical(dataY)

#build model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(Y.shape[1], activation= "softmax" ))
model.compile(loss= "categorical_crossentropy" , optimizer= "adam" , metrics=[ "accuracy" ])
model.fit(X, Y, epochs=500, batch_size=1, verbose=2)

scores = model.evaluate(X, Y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

for pattern in dataX:
    #x = numpy.reshape(pattern, (1, len(pattern, 1)))
    x = numpy.reshape(pattern, (1, 1, len(pattern)))
    x=x/float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)

