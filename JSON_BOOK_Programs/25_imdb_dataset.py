import numpy
from keras.datasets import imdb
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0) #concatinatng train and test data to use entire data as training data
Y = numpy.concatenate((Y_train, Y_test), axis=0)

print("Training data: ", X.shape,Y.shape)
print("Classes:  ", numpy.unique(Y))
print("unique words", len(numpy.unique(numpy.hstack(X))))
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()
