#Pixel Centering: scale pixel values to have a zero mean.
#Pixel Standardization: scale pixel values to have a zero mean and unit variance.
#The pixel standardization is supported at two levels: either per-image (called sample-wise) or per-dataset
# (called feature-wise). Specifically, the mean and/or mean and standard deviation statistics required to standardize
# pixel values can be calculated from the pixel values in each image only (sample-wise) or across the entire training
# dataset (feature-wise).
#here we are performing feature standardization
#mean and standard deviation is calculated from entire dataset
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)      #At this shape results are correct but showing warning
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)      #At this shape results are not correct but no warning
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
#datagen = ImageDataGenerator(zca_whitening=True)
#datagen = ImageDataGenerator(rotation_range=90)
#datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
#datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
datagen = ImageDataGenerator()
os.makedirs('images')

datagen.fit(X_train)                #fit method calculate any statistics required

#for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=9):
for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=9, save_to_dir='images', save_prefix='aug', save_format='png'):
    for i in range(0,9):
        pyplot.subplot(330+1+i)
        pyplot.imshow(X_batch[i].reshape(28,28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break
