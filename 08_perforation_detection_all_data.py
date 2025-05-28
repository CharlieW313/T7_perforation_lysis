import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

### This script calculates the perforation duration of all the qualifying events. Excluded events are described in the table '10ms_lysis_fiji_data_summary.csv'.
### The approach and method are described in detail in the script '07_perforation_duration_algorithm_testing.py'. 
### Events where the lysis (the full cell envelope breakdown) has been analysed using the script '06_lysis_detection_all_data.py' are analysed first.
### Then, the start time of lysis for events which were excluded from the full lysis analysis are estimated, and these start times are used to calculate the
### perforation duration as in the first part of the script. In the paper, the start of perforation is t4, and the start of lysis is t5. Perforation duration is t5-t4.

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


# cells = {cell_number: [trench_number, start_timepoint]}
cells = {1: [1, 0],
         2: [1, 0],
         3: [1, 0],
         4: [1, 0],
         5: [1, 10000],
         6: [1, 30000],
         7: [1, 30000],
         9: [2, 0],
         10: [2, 0],
         11: [2, 0],
         12: [2, 0],
         13: [2, 0],
         15: [2, 10000],
         16: [2, 10000],
         17: [2, 20000],
         18: [2, 20000],
         19: [2, 60000],
         20: [3, 0],
         22: [3, 0],
         23: [3, 0],
         24: [3, 0],
         25: [3, 10000],
         26: [3, 10000],
         27: [3, 10000],
         28: [3, 10000],
         29: [3, 10000],
         30: [3, 0],
         31: [3, 20000],
         32: [4, 10000],
         33: [4, 10000],
         34: [4, 10000],
         35: [4, 10000],
         36: [4, 10000],
         38: [4, 10000],
         39: [4, 10000],
         40: [4, 20000],
         41: [4, 20000],
         42: [4, 0],
         43: [5, 0],
         44: [5, 30000],
         45: [5, 60000],
         46: [5, 60000],
         47: [5, 60000],
         48: [5, 60000]}

clean = [1,2,3,4,6,7,9,10,12,13,15,16,17,18,19,23,24,25,26,27,28,29,30,31,34,35,36,38,39,40,41,42,43,44,45,47] # the inclusion list for events, clean for both perforation and lysis

# create start times dict in index space
# the start_adjust ensures that the algorithm for perforation detection starts at an appropriate time (reasoning explained at start of script '07_perforation_detection_algorithm_testing.py')
start_adjust = {1: 500,
                3: 300,
                5: 100,
                11: 200,
                16: -200,
                18: 2000,
                19: 2000,
                24: 500,
                29: -200,
                30: -200,
                39: 500,
                45: 500}

start_times = {}
for c in clean:
    t = 1000  # 1000 timepoints will correspond to approximately 10 seconds before maximal rate of contrast loss
    if c in start_adjust.keys():
        t = t + start_adjust[c]
    start_times[c] = t

# load the full lysis analysis from '06_lysis_detection_all_data.py'
envelope_breakdown = pd.read_csv("dataframes/cell_envelope_breakdown_analysis.csv")

# find the perforation start times  
slow_lysis = {}
for k, v in start_times.items():
    q = envelope_breakdown[envelope_breakdown["cell"] == k]
    peak = q["peak_time"].tolist()[0]
    lysis_t_start = q["rise_time"].tolist()[0]
    d = pd.read_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))
    timepoint = d["timepoint"][d["time"] == peak].tolist()[0]
    start_t = d["time"][d["timepoint"] == timepoint-v].tolist()[0]
    
    value_arr = np.asarray(d["c"])
    
    time_arr = np.asarray(d["time"])
    start_idx = np.where(time_arr == float(start_t))
    start_idx = start_idx[0][0]
    
    mu = np.mean(value_arr[start_idx:start_idx+200])
    std = np.std(value_arr[start_idx:start_idx+200], ddof=1)
    
    threshold_value = mu + 3*std
    window_length = 5
    rise_idx, rise = find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=start_idx, mode="increasing")
    
    slow_lysis[k] = [rise, lysis_t_start]

### There are also five events which were excluded from the lysis analysis, but were deemed suitable for perforation analysis.
### However, the calculation of perforation duration requires an end point, which is defined by the start of lysis. 
### Since the reason for excluding these data from the lysis analysis was that an adjacent cell moved into the mask position
### as the cell envelope broke down, movement of the ajacent cell into the mask position only starts after perforation ends.
### Therefore, the start of the lysis (end of perforation) can still be accurately determined.

# first find the lysis start time for these events, as in '06_lysis_detection_all_data.py'
lysis_times_adjusted = pd.read_csv("10ms_lysis_times_adjusted.csv")
slow_only = [5,11,20,22,46]
fast_lysis_slow_only = {}
for k in cells.keys():
    if k in slow_only:
        d = pd.read_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))
        lys_t = lysis_times_adjusted["lysis_t"][lysis_times_adjusted["cell"] == k].tolist()[0]
        d = d[(d["time"] >= lys_t - 2) & (d["time"] < lys_t + 2)] # gives an approximate window to work with
        
        sg = savgol_filter(d["c"], 8, 3)  # third order savgol, window size = 8
        dsg = np.concatenate((np.asarray([0]), np.diff(sg))) 
        d["dsg"] = dsg
        
        peaks, properties = find_peaks(dsg, distance=len(d["time"]))
        
        peak_idx = peaks[0]
        peak = np.asarray(d["time"])[peak_idx]
        tp_0 = peak - 1.5
        tp_1 = peak - 0.5
        mu = np.mean(d["dsg"][(d["time"] >= tp_0) & (d["time"] < tp_1)])
        std = np.std(d["dsg"][(d["time"] >= tp_0) & (d["time"] < tp_1)], ddof=1)
        
        value_arr = np.asarray(d["dsg"])
        time_arr = np.asarray(d["time"])
        threshold_value = mu + 3*std
        window_length = 5
        start_idx = peak_idx - 60 # this uses indexing and is therefore robust to the time adjustment
        rise_idx, rise = find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=start_idx, mode="increasing")
        
        fast_lysis_slow_only[k] = [rise, peak, rise_idx, peak_idx]
        
# then used the lysis start times to help find the perforation start time, as above.
start_times_slow_only = {}
for c in slow_only:
    t = 1000  
    if c in start_adjust.keys():
        t = t + start_adjust[c]
    start_times_slow_only[c] = t
    
slow_lysis_slow_only = {}
for k, v in start_times_slow_only.items():
    peak = fast_lysis_slow_only[k][1]
    lysis_t_start = fast_lysis_slow_only[k][0]
    d = pd.read_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))
    value_arr = np.asarray(d["c"])
    time_arr = np.asarray(d["time"])
    peak_idx = np.where(time_arr == float(peak))
    peak_idx = peak_idx[0][0]
    start_idx = peak_idx - v
    
    mu = np.mean(value_arr[start_idx:start_idx+200])
    std = np.std(value_arr[start_idx:start_idx+200], ddof=1)
    
    threshold_value = mu + 3*std
    window_length = 5
    rise_idx, rise = find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=start_idx, mode="increasing")
    
    slow_lysis_slow_only[k] = [rise, lysis_t_start]

# collate the two datasets
slow_lysis_all = {}
slow_lysis_all.update(slow_lysis)
slow_lysis_all.update(slow_lysis_slow_only)

# organise into a table, calculate the perforation duration and save the data
df = pd.DataFrame()
cell_ids = []
start_times = []
end_times = []
perforation_duration = []
for k in sorted(slow_lysis_all):
    cell_ids.append(k)
    start_times.append(slow_lysis_all[k][0])
    end_times.append(slow_lysis_all[k][1])
    perforation_duration.append(slow_lysis_all[k][1] - slow_lysis_all[k][0])
    
df["cell"] = cell_ids
df["start_time"] = start_times
df["end_time"] = end_times
df["perforation_duration"] = perforation_duration
    
try:
    os.mkdir("dataframes")
except:
    pass

df.to_csv("dataframes/perforation_analysis.csv")
