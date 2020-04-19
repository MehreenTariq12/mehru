#normally memory in LSTM resets after every batch, but we can manually set it not to reset for the entire training and also during model prediction and evaluation
import pandas
import matplotlib.pyplot as plt
import numpy
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM

#convert array to dataset matrix
def create_dataset(dataset, lookback =1):
    dataX, dataY = [], []
    for i in range(len(dataset)-lookback-1):
        dataX.append(dataset[i:(lookback+i),0])
        dataY.append(dataset[i+lookback,0])
    return numpy.array(dataX),numpy.array(dataY)

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("international-airline-passengers.csv", usecols=[1], header=None)
#plt.plot(dataset)
#plt.show()
dataset = dataframe.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset)*0.67)
test_size = len(dataset)-train_size

train, test = (dataset[0:train_size,:]), (dataset[train_size:len(dataset),:])
print(len(train), len(test))

look_back = 3
train_X,train_Y = create_dataset(train,look_back)
test_X,test_Y = create_dataset(test,look_back)

train_X = numpy.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))        #for time steps
test_X = numpy.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

batch_Size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_Size, look_back,1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_Size, look_back,1), stateful=True))      #for stacked LSTM
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(100):
    model.fit(train_X, train_Y, epochs=1, batch_size=batch_Size, verbose=2, shuffle=False)
    model.reset_states()
trainPredict = model.predict(train_X, batch_size=batch_Size)
testPredict = model.predict(test_X, batch_size=batch_Size)

#invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
train_Y = scaler.inverse_transform([train_Y])
testPredict = scaler.inverse_transform(testPredict)
test_Y = scaler.inverse_transform([test_Y])

trainScore = math.sqrt(mean_squared_error(train_Y[0], trainPredict[:,0]))
print( "Train Score: %.2f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(test_Y[0], testPredict[:,0]))
print( "Test Score: %.2f RMSE" % (testScore))

trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()