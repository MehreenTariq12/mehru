from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.constraints import maxnorm
import numpy

seed = 7
numpy.random.seed(seed)
(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

#print(train_X.shape, train_Y.shape) #shape of dataset is already in desired format
#print(test_X.shape, test_Y.shape)

#normalizing data
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X/255
test_X = test_X/255


train_Y = np_utils.to_categorical(train_Y)
test_Y = np_utils.to_categorical(test_Y)
num_classes = test_Y.shape[1]

#model strucure
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32,32,3), activation='relu', kernel_constraint=maxnorm(3), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu', kernel_constraint=maxnorm(3), padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#compile
l_rate=0.01
epoch = 25
decay = l_rate/epoch
sgd = SGD(learning_rate=0.01, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(train_X, train_Y,validation_data=(test_X,test_Y), epochs=epoch, batch_size=32, verbose=2)

scores = model.evaluate(test_X,test_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#for i in range(9):
 #   plt.subplot(330+i+1)
  #  plt.imshow(train_X[i])
#plt.show()