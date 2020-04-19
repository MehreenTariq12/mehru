import pandas
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
seed = 7
numpy.random.seed(seed)

#engine{‘c’, ‘python’} Parser engine to use. The C engine is faster while the python engine is currently more feature-complete
#Number of lines at bottom of file to skip (Unsupported with engine=’c’)

dataframe = pandas.read_csv("international-airline-passengers.csv", usecols=[1], header=None)
#plt.plot(dataset)
#plt.show()
dataset = dataframe.values
dataset = dataset.astype('float32')

train_size = int(len(dataset)*0.67)
test_size = len(dataset)-train_size

train, test = (dataset[0:train_size,:]), (dataset[train_size:len(dataset),:])
print(len(train), len(test))

#convert array to dataset matrix
def create_dataset(dataset, lookback =1):
    dataX, dataY = [], []
    for i in range(len(dataset)-lookback-1):
        dataX.append(dataset[i:(lookback+i),0])
        dataY.append(dataset[i+lookback,0])
    return numpy.array(dataX),numpy.array(dataY)
#look_back = 1
look_back=10
train_X,train_Y = create_dataset(train,look_back)
test_X,test_Y = create_dataset(test,look_back)

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=200, batch_size=2, verbose=2)
trainScore = model.evaluate(train_X,train_Y,verbose=0)
print("Train Score: %.2f MSE (%.2f RMSE)" % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(test_X,test_Y,verbose=0)
print("Test Score: %.2f MSE (%.2f RMSE)" % (testScore, math.sqrt(testScore)))
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()