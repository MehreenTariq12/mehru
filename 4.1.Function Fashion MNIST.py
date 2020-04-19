import tensorflow as tf
import tensorflow_datasets as tfds

#import helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

#improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

class FMNIST():
    def __init__(self):
        self.dataset, self.metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
        self.train_dataset, self.test_dataset = self.dataset['train'], self.dataset['test']
        self.num_train_examples = self.metadata.splits['train'].num_examples
        self.num_test_examples = self.metadata.splits['test'].num_examples

    def PreProcess(self, images, labels):
        images = tf.cast(images, tf.float32)
        images /= 255
        return images, labels

    def generateModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, BATCH_SIZE, model, epochs):
        self.train_dataset = self.train_dataset.repeat().shuffle(self.num_train_examples).batch(BATCH_SIZE)
        history = model.fit(self.train_dataset, epochs=epochs, steps_per_epoch=math.ceil(self.num_train_examples / BATCH_SIZE))
        return history

    def test(self, BATCH_SIZE, model):
        self.test_dataset = self.test_dataset.batch(BATCH_SIZE)
        test_loss, test_accuracy = model.evaluate(self.test_dataset, steps=math.ceil(self.num_test_examples / BATCH_SIZE))
        return test_loss,test_accuracy

    def display_single_img(self,index):
        for image, label in self.test_dataset.take(index):
            break
        image = image.numpy().reshape((28, 28))
        plt.figure()
        plt.imshow(image, cmap=plt.cm.binary)
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def plot_results(self, history):
        plt.xlabel("Epochs numbers")
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'])
        plt.show()
        plt.xlabel("Epochs numbers")
        plt.ylabel("Accuracy Magnitude")
        plt.plot(history.history['accuracy'])
        plt.show()


def main():
    object1 = FMNIST()
    BATCH_SIZE = 32
    epochs = 10
    object1.train_dataset = object1.train_dataset.map(object1.PreProcess)
    object1.test_dataset = object1.test_dataset.map(object1.PreProcess)
    model = object1.generateModel()
    history = object1.train(BATCH_SIZE, model, epochs)
    object1.plot_results(history)
    test_loss, test_accuracy = object1.test(BATCH_SIZE, model)
    print("Testing accuracy =",test_accuracy)
    print("Testing loss = ",  test_loss)
    #object1.display_single_img(1)

main()




