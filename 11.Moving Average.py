import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
series = baseline + trend(time, 0.1)

plt.figure(figsize = (10, 6))
plot_series(time, series)

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 *np.pi), 1/np.exp(3 *season_time))

def seasonality(time, period, amplitude = 1, phase = 0):
    season_time = ((time +phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

amplitude = 40
series = seasonality(time, period=365, amplitude= amplitude)
plt.figure(figsize = (10, 6))
plot_series(time, series)

slope = 0.05
series = baseline +trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude )
plt.figure(figsize = (10, 6))
plot_series(time, series)
def white_noise(time, noise_level = 1, seed = None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed = 42)
plt.figure(figsize = (10, 6))
plot_series(time, noise)

series += noise
plt.figure(figsize=(10, 6))
plot_series(time, series)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
#we want to forecast just on validation period
naive_forecast = series[split_time-1:-1]        # copythe value of previouse day

plt.figure(figsize = (10, 6))
plot_series(time_valid, x_valid, label = "Series")
plot_series(time_valid, naive_forecast, label = "forecast")


plt.figure(figsize = (10, 6))
plot_series(time_valid, x_valid, start = 0, end = 150, label = "series")
plot_series(time_valid, naive_forecast, start = 1, end = 151, label = "forecast")
plt.show()

errors = naive_forecast -x_valid
abs_errors = np.abs(errors)     #computing absolute of errors
mae = abs_errors.mean()     #computing mean of absolute errors
print(mae)

mae = tf.keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()
print(mae)

#Moving Average
#Key idea, tomorrow will be cause by the average of past few days values
def moving_average_forecast(series, window_size):
    "Forecast mean of the past few values, If windowsize = 1 it would be equal to naive forecast"
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append((series[time:time + window_size].mean()))
    return np.array(forecast)

def moving_average_forecast2(series, window_size):
    mov = np.cumsum(series)
    mov[window_size:] = mov[window_size:] - mov[:-window_size]
    return mov[window_size-1: -1]/window_size

moving_average = moving_average_forecast2(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="series")
plot_series(time_valid, moving_average, label="Moving average (30 days)")
plt.show()

mae = tf.keras.metrics.mean_absolute_error(x_valid, moving_average).numpy
print(mae)

# results are worse
# now we will do differencing
# it will remoove trend and seasonality

diff_series = (series[365:] - series[:-365])
diif_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diif_time, diff_series, label = "Series(t) - Series(t - 365)")

# zooming in validation period
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label = "Series(t) - Series(t-365)")

diff_moving_Avg = moving_average_forecast(diff_series, 50)[split_time-365-50:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label = "Series(t) - Series(t-365)")
plot_series(time_valid, diff_moving_Avg, label="Moving average of difference")


diff_moving_Avg_plus_past = series[split_time - 365:-365] + diff_moving_Avg
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_Avg_plus_past, label="Forecasts")

mae = tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_Avg_plus_past).numpy
print(mae)

diff_moving_Avg_plus_smoot_past = moving_average_forecast(series[split_time-370:-359], 11) +diff_moving_Avg
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_Avg_plus_smoot_past, label="Forecasts")
plt.show()

mae = tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_Avg_plus_smoot_past).numpy
print(mae)

