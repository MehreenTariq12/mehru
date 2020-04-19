import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)
for i,c in enumerate(celsius_q):
    print("{} degree celsius = {} degree fahrenheit".format(c, fahrenheit_a[i]))
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])                                        #units isthe number of input variables and , input shape is the number of inputs
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))           # tf using these parameters during trainingto find the best model , 0.1 here is the learning rate
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)                     #training is done here, 1 epoch is the full iteration of the full examples we have.
plt.xlabel("Epochs numbers")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
print(model.predict([100.0]))
print("These are the layer variables: {}".format(l0.get_weights()))
cv2.waitKey(0)
