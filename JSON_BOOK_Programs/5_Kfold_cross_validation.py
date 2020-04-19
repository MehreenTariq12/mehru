from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
seed = 7
numpy.random.seed(seed)
#load data
dataset = numpy.loadtxt("diabetes.csv", delimiter=",")
#split in attributes and class
X = dataset[:, 0:8]
Y = dataset[:, 8]
#define 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test  in kfold.split(X,Y):
    #create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu' ))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    #Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Fit model
    model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
    #evaluate model
    scores = model.evaluate(X[test],Y[test], verbose=0)
    print("%s:   %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))