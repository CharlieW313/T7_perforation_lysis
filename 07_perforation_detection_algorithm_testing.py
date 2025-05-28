import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

### This script details the algorithm development for perforation detection (the steady loss of material from the cell between times t4 and t5). 
### The algorithm works by first finding the mean and standard deviation of the phase contrast intensity of the cell over a 200 time point window (approximately 2 seconds),
### where the window begins, in this case, 1500 time points before the peak rate of intensity change as found in the script '06_lysis_detection_all_data.py'.
### Since the perforation period differs between cells, and because the masks are static but the cells sometimes move in the trench, the window to calculate the mean
### and standard deviation must be taken shortly before the perforation begins, to ensure it truly reflects the contrast of the cell prior to perforation. 
### If not chosen carefully, the background window could erroneously reflect the contrast of a different cell or trench background if cells move, 
### or the contrast during the perforation itself if the background window starts too late. 
### For these reasons the start time of the background window for each cell is detailed in the next script. 
### The perforation start is declared when the phase contrast intensity increases above the calculated mean plus 3 standard deviations, for a minimum of 5 consecutive time points.

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

k = 1 # use cell ID = 1 as an example lysis
envelope_breakdown = pd.read_csv("dataframes/cell_envelope_breakdown_analysis.csv")
q = envelope_breakdown[envelope_breakdown["cell"] == k]
peak = q["peak_time"].tolist()[0]
lysis_t_start = q["rise_time"].tolist()[0]
d = pd.read_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))

# find starting time index, 1500 timepoints before peak
# find peak index -> find time point -> subtract 1500 -> find corresponding time
timepoint = d["timepoint"][d["time"] == peak].tolist()[0]
start_t = d["time"][d["timepoint"] == timepoint-1500].tolist()[0]

# the raw data is not as noisy as its derivative, so no need to use Savitzky-Golay filters for this analysis
value_arr = np.asarray(d["c"])
time_arr = np.asarray(d["time"])
start_idx = np.where(time_arr == float(start_t))
start_idx = start_idx[0][0]
peak_idx = np.where(time_arr == float(peak))
peak_idx = peak_idx[0][0]

# find mean and standard deviation in 200 timepoint window
mu = np.mean(value_arr[start_idx:start_idx+200])
std = np.std(value_arr[start_idx:start_idx+200], ddof=1)

# find index and time when the intensity rises above the threshold, i.e. when material begins to leak out of the cell. This is the perforation start.
threshold_value = mu + 3*std
window_length = 5
rise_idx, rise = find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=start_idx, mode="increasing")

# example plot with key time points and time windows indicated
plt.figure(figsize=(20,8))
plt.plot(time_arr, value_arr, label="Phase contrast intensity")
plt.vlines([time_arr[start_idx], time_arr[start_idx+200]], np.min(value_arr[start_idx:peak_idx]), np.max(value_arr[start_idx:peak_idx]), linestyle = "--", color="k", label="Window for threshold calculation")
plt.vlines([lysis_t_start], np.min(value_arr[start_idx:peak_idx]), np.max(value_arr[start_idx:peak_idx]), linestyle = "--", color="g", label="Lysis start time (t5 in paper)")
plt.vlines([rise], np.min(value_arr[start_idx:peak_idx]), np.max(value_arr[start_idx:peak_idx]), linestyle = "--", color="m", label="Perforation start time (t4 in paper)")
plt.hlines([mu - 3*std, mu + 3*std], time_arr[start_idx], time_arr[peak_idx], linestyle="-.", color="k", label="+/- 3 standard deviations around the mean in threshold window")
plt.xlim([start_t - 1, peak+1])
plt.ylim([290, 390])
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Phase contrast intensity (AU/px)", fontsize=16)
plt.legend(fontsize=14, loc=2)
plt.show()
plt.close()