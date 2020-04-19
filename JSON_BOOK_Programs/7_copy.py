from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
def create_model(optimizer='rmsprop', init='glorot-uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, init=init, activation='relu'))
    model.add(Dense(8, init=init, activation='relu'))
    model.add(Dense(1, init=init, activation='sigmoid'))
    #compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("diabetes.csv", delimiter=",")
X= dataset[:, 0:8]
Y = dataset[:, 8]

model = KerasClassifier(build_fn=create_model, verbose=0)

optimizers= ['rmsprop', 'adam']
epochs= [50, 100, 150]
batches= [5, 10, 20]
init= ['normal', 'uniform', 'glorot_uniform']

param_grid = dict(optimizer=optimizers, batch_size = batches, nb_epoch=epochs, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X,Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))