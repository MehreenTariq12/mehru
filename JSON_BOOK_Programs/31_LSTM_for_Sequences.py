import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout

seed = 7
numpy.random.seed(seed)

top_words = 5000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32
model = Sequential()
#1
#model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

'''
#2
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length, dropout=0.2))        #for dropout
model.add(Dropout(0.2)) #for dropout
model.add(LSTM(100))
model.add(Dropout(0.2)) #for dropout
model.add(Dense(1, activation='sigmoid'))
'''

#ALternative dropoutway
# the dropout W for conﬁguring the input dropout and dropout U for conﬁguring the recurrent dropout
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length, dropout=0.2))        #for dropout
model.add(LSTM(100, dropout_U= 0.2, dropout_W=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=3, batch_size=64, )
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
