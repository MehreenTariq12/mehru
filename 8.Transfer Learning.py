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

grace_hopper = tf.keras.utils.get_file('image.jpg', 'http://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES,IMAGE_RES))
grace_hopper = np.array(grace_hopper)/255.0
print(grace_hopper.shape)
plt.imshow(grace_hopper)
plt.show()

result = model.predict(grace_hopper[np.newaxis, ...])           #Model always need a batch of images to process. So we add a batch dimension and pass the image for prediction to the model.

print(result.shape)

predicted_class = np.argmax(result[0], axis = -1)
print(predicted_class)

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','http://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title('Prediction: ' + predicted_class_name.title())
plt.show()
t = time.time()
export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)
model.save(export_path_keras)
