import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

### This script details the algorithm development for lysis detection (the breakdown of the cell envelope). Variables referring to fast_lysis are referring to this process.
### The detection algorithm will be based on finding the time point at which the time derivative of phase contrast intensity rapidly accelerates; in practice this will be
### detected by determining when it crosses a threshold determined by the mean and variance of the time derivative over the preceding period. 
### The first step will be to develop a suitable filter, as the time derivative of the unfiltered phase contrast intensity is a noisy signal.

# load estimated lysis times
lysis_times_adjusted = pd.read_csv("10ms_lysis_times_adjusted.csv")

test_cell = 6
d = pd.read_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(test_cell).zfill(2))) # time series intensity data
lys_t = lysis_times_adjusted["lysis_t"][lysis_times_adjusted["cell"] == test_cell].tolist()[0]
d = d[(d["time"] >= lys_t - 2) & (d["time"] < lys_t + 2)]  # observe a period 2 seconds before and after the estimated lysis time.

# compare two filtering algorithms, moving average and Savitzky-Golay
# moving average implementation
def moving_average(series, window_size=5):
    series = np.asarray(series)
    averaged = np.zeros(series.shape)
    for count, value in enumerate(series[window_size-1:]):
        averaged[count+window_size-1] = np.sum(series[count:count+window_size])/window_size
    averaged[0:window_size-1] = np.nan
    return averaged

# derivative of unfiltered signal
dc = np.concatenate((np.asarray([0]), np.diff(d["c"]))) # a zero time point appended to the front of the array so that it has equal length to and can be plotted with the intensity data

# derivative of moving average filtered signal
ma = moving_average(d["c"])
dma = np.concatenate((np.asarray([0]), np.diff(ma)))

# derivative of Savitzky-Golay filtered signal, using window size of 8 data points and a third order polynomial.
sg = savgol_filter(d["c"], 8, 3)
dsg = np.concatenate((np.asarray([0]), np.diff(sg)))

### Plot the result. Notice that while moving average and Savitzky-Golay filters give similar noise reduction, the moving average introduces an undesirable time shift.
### Savitzky-Golay does not bias the time, as can be seen by the peak matching well to the unfiltered peak.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
ax.plot(d["time"], d["c"], color="b", label="Intensity")
ax2 = ax.twinx()
ax2.plot(d["time"], dc, color="r", label="Derivative")
ax2.plot(d["time"], dsg, color="k", label="Derivative of Savitsky-Golay filtered trace")
ax2.plot(d["time"], dma, color="gold", label="Derivative of moving average filtered trace")
ax.set_xlabel("Time (s)", fontsize=16)
ax.set_ylabel("Intensity (AU/px)", fontsize=16)
ax2.set_ylabel("dI/dt (AU px^-1 timepoint^-1)", fontsize=16)
ax.legend(fontsize=14)
ax2.legend(fontsize=14)
plt.xlim([lys_t-0.2,lys_t+0.8])
plt.show()
plt.close()

### Using the derivative of the Savitzky-Golay filtered signal, we now aim to find the times at which the derivative reaches a maximum, 
### when it crosses the rise threshold (the mean plus three standard deviations of the filtered derivative between 1.5 and 0.5 seconds before the peak),
### and when the rate derivative falls below the fall threshold (the half-maximal value of the filtered derivative).
peaks, properties = find_peaks(dsg, distance=len(d["time"]))
peak_idx = peaks[0] # index of peak
t_peak = np.asarray(d["time"])[peak_idx] # time of peak
tp_0 = t_peak - 1.5
tp_1 = t_peak - 0.5
d["dsg"] = dsg

# find mean and standard deviation of derivative of phase contrast intensity between 1.5 and 0.5 seconds before peak.
mu = np.mean(d["dsg"][(d["time"] >= tp_0) & (d["time"] < tp_1)])
std = np.std(d["dsg"][(d["time"] >= tp_0) & (d["time"] < tp_1)], ddof=1)

# example plot. The black dashed horizontal lines are +/- 3 standard deviations around the mean of the derivative. 
# the red vertical dashed lines indicate 1.5 and 0.5 seconds before the peak. This indicates the time period for the mean and standard deviation calculation.
# the magenta vertical dashed line indicates the peak, the maximum derivative. 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
ax.plot(d["time"], dsg, label="derivative of S-G filtered intensity")
ax.vlines([tp_0, tp_1], np.min(dsg), np.max(dsg), color="r", linestyle="--", label="Period for threshold calculation")
ax.vlines([np.asarray(d["time"])[peak_idx]], np.min(dsg), np.max(dsg), color="m", linestyle="--", label="Derivative maximum")
ax.hlines([mu-3*std, mu+3*std], tp_0, tp_1+2, color="k", linestyle="--", label="+/- 3 std devs of the mean")
ax.set_ylim([mu-4*std, mu+4*std])
ax.set_ylabel("dI/dt (AU px^-1 timepoint^-1)", fontsize=16)
ax.set_xlabel("Time (s)", fontsize=16)
ax.legend(fontsize=14, loc=4)
plt.show()
plt.close()

### Now we need to find the point at which the derivative crosses the threshold (mean + 3 standard deviations)

# define a function that can find when time series crosses a defined threshold
def find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=0, mode="increasing"):
    """
    A general function for finding when a time series (value_arr) crosses threshold_value for a minimum of
    window_length number of time points. The starting index is specified by start_idx if error causing or
    irrelevant data in the array needs to be skipped over. The mode can be set to "increasing" (default) if
    you wish to find when an array increases above a threshold, or set to a different value (e.g "decreasing")
    if you wish to find when it decreases below a threshold. 
    
    return: crossing_idx, the first index at which the array crosses the threshold for a minimum of window_length 
    consecutive time points. Also returns the corresponding time.
    If the function reaches the end of the array without meeting the threshold crossing conditions, return None.
    """
    threshold_counter = 0
    idx = start_idx - 1
    while (idx < len(value_arr)) and (threshold_counter < window_length):
        idx = idx + 1
        value = value_arr[idx]
        t = time_arr[idx]
        if mode == "increasing":
            if value > threshold_value:
                threshold_counter = threshold_counter + 1
            else:
                threshold_counter = 0
        else:
            if value < threshold_value:
                threshold_counter = threshold_counter + 1
            else:
                threshold_counter = 0
    
    if idx < len(value_arr):
        crossing_idx = idx - (window_length - 1)
        return crossing_idx, time_arr[crossing_idx]
    else:
        return None

# find when the lysis starts, i.e when the derivative rises above the threshold of 3 standard deviations above the mean derivative during perforation
# this is the time reported as t5 in the paper.
value_arr = np.asarray(d["dsg"])
time_arr = np.asarray(d["time"])
threshold_value = mu + 3*std
window_length = 3
start_idx = peak_idx - 30
rise_idx, rise = find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=start_idx, mode="increasing")

# for interest, find when the rate of material loss substantially decreases. These values are calculated but not used in the paper.
threshold_value = value_arr[peak_idx] * 0.5
window_length = 3
start_idx = peak_idx
fall_idx, fall = find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=start_idx, mode="decreasing")   

# Plot indicating method for calculating the time of lysis (t5 in paper, point of crossing rise threshold), 
# the maximum rate of loss (not used in paper, for interest), and the point at which the rate of loss falls to its half maximal rate (not used in paper, for interest)
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(d["time"], dsg, marker="x", label="derivative of S-G filtered intensity")
ax.vlines([np.asarray(d["time"])[peak_idx]], np.min(dsg), np.max(dsg), color="m", linestyle="--", label="Derivative maximum")
ax.vlines([np.asarray(d["time"])[rise_idx], tp_0, tp_1], np.min(dsg), np.max(dsg), color="r", linestyle="--", label="Period for threshold calculation and point of crossing threshold (t5 in paper)")
ax.vlines(np.asarray(d["time"])[fall_idx], np.min(dsg), np.max(dsg), color="b", linestyle="--", label="Point of crossing fall threshold")
ax.hlines([value_arr[peak_idx] * 0.5], tp_0, tp_1+2, color="b", linestyle="--", label="Half of the maximum (peak) derivative")
ax.hlines([mu-3*std, mu+3*std], tp_0, tp_1+2, color="k", linestyle="--", label="+/- 3 std devs of the mean")
ax.set_xlim([np.asarray(d["time"])[peak_idx]-2, np.asarray(d["time"])[peak_idx]+0.2])
ax.set_ylabel("dI/dt (AU px^-1 timepoint^-1)", fontsize=16)
ax.set_xlabel("Time (s)", fontsize=16)
ax.legend(fontsize=14, loc=2)
plt.show()
plt.close()
