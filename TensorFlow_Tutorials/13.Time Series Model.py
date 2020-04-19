import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format = "-", start = 0,  end  = None, label= None):
    plt.plot(time[start:end], series[start:end], format, label = label)
    plt.xlabel = ("Time")
    plt.ylabel = ("Value")
    if label:
        plt.legend(fontsize = 14)
        plt.grid(True)

def trend(time, slope = 0):
    return slope*time

time = np.arange(4 * 365 + 1)
baseline = 10
series = baseline + trend(time, 20.1)

#plt.figure(figsize = (10, 6))
#plot_series(time, series)
#plt.show()

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 *np.pi), 1/np.exp(3 *season_time))

def seasonality(time, period, amplitude = 1, phase = 0):
    season_time = ((time +phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

amplitude = 40
series = seasonality(time, period=365, amplitude= amplitude)
#plt.figure(figsize = (10, 6))
#plot_series(time, series)
#plt.show()

slope = 0.05
series = baseline +trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude )
#plt.figure(figsize = (10, 6))
#plot_series(time, series)
#plt.show()
def white_noise(time, noise_level = 1, seed = None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed = 42)
#plt.figure(figsize = (10, 6))
#plot_series(time, noise)
#plt.show()

series += noise
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

def window_dataset(series, window_size, batch_size = 32, shuffle_buffer = 1000 ):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(2).prefetch(1)
    return dataset

tf.random.set_seed(42)
np.random.seed(42)
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape = [window_size])
])

optimizer = tf.keras.optimizers.SGD(lr = 1e-5, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
model.fit(train_set, epochs = 100, validation_data = valid_set)

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape = [window_size])
])
lr_shedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-6 * 10**(epochs/30))
optimizer = tf.keras.optimizers.SGD(lr = 1e-5, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
history= model.fit(train_set, epochs = 100, callbacks = [lr_shedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 1e-3, 0, 20])
plt.show()

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape = [window_size])
])
optimizer = tf.keras.optimizers.SGD(lr = 1e-5, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10)

model.fit(train_set, epochs = 500, validation_data = valid_set, callbacks = [early_stopping])

#Lets make our model to make predictions
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

lin_forecast = model_forecast(model, series[split_time-window_size:-1], window_size)[:, 0]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, lin_forecast)
plt.show()

#deep model
window_size = 30
train_set = window_dataset(x_train, window_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation = "relu", input_shape = [window_size]),
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(1)
])
lr_shedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-7 * 10**(epochs/30))
optimizer = tf.keras.optimizers.SGD(lr = 1e-7, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
history= model.fit(train_set, epochs = 100, callbacks = [lr_shedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-7, 5e-3, 0, 30])
plt.show()

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.SGD(lr = 1e-5, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10)

model.fit(train_set, epochs = 500, validation_data = valid_set, callbacks = [early_stopping])

#Lets make our model to make predictions
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

lin_forecast = model_forecast(model, series[split_time-window_size:-1], window_size)[:, 0]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, lin_forecast)
plt.show()
#make prediction on the validation period
dense_forcast = model_forecast(
    model, series[split_time - window_size:-1],
    window_size)[:, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series((time_valid, dense_forcast))

print(tf.keras.metrics.mean_absolute_error(x_valid, dense_forcast).numpy())