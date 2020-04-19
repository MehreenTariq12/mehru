from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import cv2
import time
CLASSIFIER_URL = "http://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_RES = 224
model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape = (IMAGE_RES,IMAGE_RES, 3))
])

import numpy as np
import PIL.Image as Image

splits = tfds.Split.ALL.subsplit(weighted=(80, 20))
splits, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split= splits)
(train_examples, validation_examples) = splits
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','http://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

for i, example_image in enumerate(train_examples.take(3)):
    print('Image {} shape: {}'.format(i+1, example_image[0].shape))

def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES,IMAGE_RES))/255.0
    return image, label
BATCH_SIZE = 32
train_batches = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()
result_batch = model.predict(image_batch)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
    _ = plt.suptitle('Imagenet prediction')
plt.show()

# Tensorflow hub is used for calling partial model, this model will take input do all processings to extract feature of image except producing the probability of the output class
URL = "http://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
feature_extractor = hub.KerasLayer(URL, input_shape = (IMAGE_RES, IMAGE_RES, 3))

feature_batch = feature_extractor(image_batch)
print(feature_batch.shape)

feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(2, activation = 'softmax')
])
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()                                                                             #chk model summary

epochs = 6
history = model.fit(train_batches, epochs = epochs, validation_data = validation_batches)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(0,0))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and validation Loss')
plt.show()

class_names = np.array(info.features['label'].names)
print(class_names)

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()                   #Removes dimensions of size 1 from the shape of a tensor
predicted_ids = np.argmax(predicted_batch, axis = -1)
predicted_class_names = class_names[predicted_ids]
print(predicted_class_names)

print("Labels:", label_batch)
print("Predicted labels", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6, 5, [n])
    color = 'blue' if predicted_ids[n]== label_batch[n] else"red"
    plt.title(predicted_class_names[n].title(), color = color)
    plt.axis('off')
    _ = plt.suptitle("Model prediction (blue: correct, red: incorrect)")
    plt.imshow
