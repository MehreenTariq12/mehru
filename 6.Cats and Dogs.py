from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

_URL = 'http://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin = _URL, extract = True)

#zip_dir_base = os.path.dirname(zip_dir)
#:find $zip_dir_base -type d -print()

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))      #Python os.listdir() Method - Python method listdir() returns a list containing the names of the entries in the directory given by path.
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("Total training cat images = ", num_cats_tr)
print("Total training dog images = ", num_dogs_tr)
print("Total validation cat images = ", num_cats_val)
print("Total validation dog images = ", num_cats_val)
print("Total training cat images = ", total_train)
print("Total validation cat images = ", total_val)

BATCH_SIZE = 100
IMG_SHAPE = 150

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size= BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size = (IMG_SHAPE,IMG_SHAPE),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='binary')
print(train_data_gen)
#Visualizing Trianing Images
#We can visualize our training imagesby getting a batch of images from training generator and thenplotting a few of them using matplotlib

sample_training_images, _ = next(train_data_gen)            # next function returns a batch from the dataset . One batch is the tuple of (many images, many labels). for right now we are discarding the labels because we just want to look at the images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize = (20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        plt.tight_layout()
        plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()                                                                             #chk model summary

EPOCHS = 10
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train/float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data = val_data_gen,
    validation_steps=int(np.ceil(total_val/float(BATCH_SIZE)))
)