import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel = ("Time")
    plt.ylabel = ("Value")
    if label:
        plt.legend(fontsize=14)
        plt.grid(True)


def trend(time, slope=0):
    return slope * time


time = np.arange(4 * 365 + 1)
baseline = 10
series = baseline + trend(time, 20.1)


# plt.figure(figsize = (10, 6))
# plot_series(time, series)
# plt.show()

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)
# plt.figure(figsize = (10, 6))
# plot_series(time, series)
# plt.show()

slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)


# plt.figure(figsize = (10, 6))
# plot_series(time, series)
# plt.show()
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


noise_level = 5
noise = white_noise(time, noise_level, seed=42)
# plt.figure(figsize = (10, 6))
# plot_series(time, noise)
# plt.show()

series += noise
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


def window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


tf.random.set_seed(42)
np.random.seed(42)
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda  x: tf.expand_dims(x, axis = -1),
    input_shape = [None]),                                          #inputshape = none means this particular model accepts inputs of any size
    tf.keras.layers.SimpleRNN(100, return_sequences = True),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda  x: x * 200.0)
])

window_size = 30
train_set = window_dataset(x_train, window_size, batch_size=128)
lr_shedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-7 * 10**(epochs/20))
optimizer = tf.keras.optimizers.SGD(lr = 1e-7, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
history= model.fit(train_set, epochs = 100, callbacks = [lr_shedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-7, 1e-3, 0, 30])
plt.show()

window_size = 30
train_set = window_dataset(x_train, window_size, batch_size=128)
valid_set = window_dataset(x_valid, window_size, batch_size=128)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda  x: tf.expand_dims(x, axis = -1),
    input_shape = [None]),                                          #inputshape = none means this particular model accepts inputs of any size
    tf.keras.layers.SimpleRNN(100, return_sequences = True),
    tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda  x: x * 200.0)
])
optimizer = tf.keras.optimizers.SGD(lr = 1.5e-6, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 50)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("my_checkpoint.h5", save_best_only = True)            #save model at every epoch
model.fit(train_set, epochs = 500, validation_data = valid_set, callbacks = [early_stopping, model_checkpoint])

model = tf.keras.models.load_model("my_checkpoint.h5")

run_forecast = model_forecast(model,
                              series[split_time - window_size: -1],
                              window_size)[:, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid,  x_valid)
plot_series(time_valid, run_forecast)

mae = tf.keras.metrics.mean_absolute_error(x_valid, run_forecast).numpy
print("first validation error without sequence", mae)
def seq2seq_window_dataset(series, window_size, batch_size = 32, shuffle_buffer = 1000):
    series = tf.expand_dims(series, axis = -1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
for x_batch, y_batch in seq2seq_window_dataset(tf.range(10),3, batch_size=1):
    print ("X:", x_batch.numpy())
    print("Y:", y_batch.numpy())

tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)


model = tf.keras.models.Sequential([            #inputshape = none means this particular model accepts inputs of any size
    tf.keras.layers.SimpleRNN(100, return_sequences = True,input_shape = [None, 1]),
    tf.keras.layers.SimpleRNN(100, return_sequences = True),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda  x: x * 200.0)
])

lr_shedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-7 * 10**(epochs/30))
optimizer = tf.keras.optimizers.SGD(lr = 1e-7, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
history= model.fit(train_set, epochs = 100, callbacks = [lr_shedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-7, 1e-4, 0, 30])
plt.show()


window_size = 30
train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(100, return_sequences = True, input_shape = [None, 1]),
    tf.keras.layers.SimpleRNN(100, return_sequences = True),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda  x: x * 200.0)
])
optimizer = tf.keras.optimizers.SGD(lr = 1e-6, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10)
model.fit(train_set, epochs = 500, validation_data = valid_set, callbacks = [early_stopping])

run_forecast = model_forecast(model,
                              series[..., np.newaxis],
                              window_size)
run_forecast = run_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid,  x_valid)
plot_series(time_valid, run_forecast)
plt.show(  )
mae = tf.keras.metrics.mean_absolute_error(x_valid, run_forecast).numpy()
print(mae)

