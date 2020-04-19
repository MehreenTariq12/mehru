import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)
for i,c in enumerate(celsius_q):
    print("{} degree celsius = {} degree fahrenheit".format(c, fahrenheit_a[i]))
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])                                        #units isthe number of input variables and , input shape is the number of inputs
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))           # tf using these parameters during trainingto find the best model , 0.1 here is the learning rate
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)                     #training is done here, 1 epoch is the full iteration of the full examples we have.
print("model finished its training")                                                        #This optimization process is called Gradient Descent. The specific algorithm used to calculate the new value of each internal variable is specified by the optimizer parameter when calling model.compile(...).
print(model.predict([100.0]))
print("model predicts for 100 degree celsius is: {} degree fahrenheit".format(model.predict([100.0])))
print("These are the l0 layer variables: {}".format(l0.get_weights()))
print("These are the l1 layer variables: {}".format(l1.get_weights()))
print("These are the l2 layer variables: {}".format(l2.get_weights()))
plt.xlabel("Epochs numbers")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
cv2.waitKey(0)
