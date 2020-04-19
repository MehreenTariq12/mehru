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
series = baseline + trend(time, 0.1)

plt.figure(figsize = (10, 6))
plot_series(time, series)
plt.show()

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 *np.pi), 1/np.exp(3 *season_time))

def seasonality(time, period, amplitude = 1, phase = 0):
    season_time = ((time +phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

amplitude = 40
series = seasonality(time, period=365, amplitude= amplitude)
plt.figure(figsize = (10, 6))
plot_series(time, series)
plt.show()

slope = 0.05
series = baseline +trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude )
plt.figure(figsize = (10, 6))
plot_series(time, series)
plt.show()
def white_noise(time, noise_level = 1, seed = None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed = 42)
plt.figure(figsize = (10, 6))
plot_series(time, noise)
plt.show()

series += noise
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

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
plt.show()

plt.figure(figsize = (10, 6))
plot_series(time_valid, x_valid, start = 0, end = 150, label = "series")
plot_series(time_valid, naive_forecast, start = 1, end = 151, label = "forecast")
plt.show()

errors = naive_forecast -x_valid
abs_errors = np.abs(errors)     #computing absolute of errors
mae = abs_errors.mean()     #computing mean of absolute errors
print(mae)