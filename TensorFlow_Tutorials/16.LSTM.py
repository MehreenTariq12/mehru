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

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

def sequential_window_dataset(series, window_Size):         #we want each batch to be consecutive, we will not do shuffling
    series = tf.expand_dims(series, axis = -1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_Size+1, shift = window_Size, drop_remainder = True)
    ds = ds.flat_map(lambda  window: window.batch(window_Size+1))
    ds = ds.map(lambda  window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)

# To demonstrate we have created a little dataset with window size 3
for x_batch, y_batch in sequential_window_dataset(tf.range(10), 3):
    print(x_batch.numpy(), y_batch.numpy())

#now we have to reset the model at the begining of every epoch
class ResetStateCallBack(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, return_sequences = True, stateful = True ,batch_input_shape = [1, None, 1]),
    tf.keras.layers.LSTM(100, return_sequences = True, stateful = True),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda  x: x * 200.0)
])
window_size = 30
train_set = sequential_window_dataset(x_train, window_size)
lr_shedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-8 * 10**(epochs/20))
optimizer = tf.keras.optimizers.SGD(lr = 1e-7, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
history= model.fit(train_set, epochs = 100, callbacks = [lr_shedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 30])
plt.show()

window_size = 30
train_set = sequential_window_dataset(x_train, window_size)
valid_set = sequential_window_dataset(x_valid, window_size)


model = tf.keras.models.Sequential([            #inputshape = none means this particular model accepts inputs of any size
    tf.keras.layers.LSTM(100, return_sequences = True, stateful = True ,batch_input_shape = [1, None, 1]),             #in shape starting 1 is representing window size 1
    tf.keras.layers.LSTM(100, return_sequences = True, stateful = True),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda  x: x * 200.0)
])

#lr_shedule = tf.keras.callbacks.LearningRateScheduler(
 #   lambda epochs: 1e-8 * 10**(epochs/30))
optimizer = tf.keras.optimizers.SGD(lr = 1e-7, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metric = ['mae'])
reset_states = ResetStateCallBack()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("my_CheckPoint.h5", save_best_only = True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 50)
history= model.fit(train_set, epochs = 500,validation_data = valid_set, callbacks = [early_stopping, model_checkpoint, reset_states])

model = tf.keras.models.load_model("my_checkpoint.h5")

#model.reset_states()
rnn_forecast = model.predict(series[np.newaxis, :, np.newaxis])         #[1, no of time steps, 1]
rnn_forecast = rnn_forecast[0, split_time-1:-1, 0]
print(rnn_forecast.shape)

plt.figure(figsize=[10, 6])
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
print(tf.keras.metrics.mean_absolute_error(x_valid,rnn_forecast).numpy())