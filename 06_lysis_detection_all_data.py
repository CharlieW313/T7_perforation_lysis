import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

### This script uses the method developed in '05_lysis_detection_algorithm_testing.py' to iterate over all included events 
### (see '10ms_lysis_fiji_data_summary.csv' for exclusions in the included_in_lysis_data column). 
### The output is a table 'cell_envelope_breakdown_analysis.csv', which details the start time of lysis (t5 in the paper, column rise_time in the table),
### in addition to the time when the rate of phase contrast intensity change reaches a maximum (proxy of maximal rate of material loss from the cell, column peak_time in the table),
### and also the time when the rate of phase contrast intensity change falls to its half maximal rate (column fall_time in table)
### Note that the columns peak_time and fall_time in 'cell_envelope_breakdown_analysis.csv' are not used in detail in the paper, and are included for interest only.

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
fast_lysis_only = [33,48] # events where slow lysis excluded but fast lysis included
clean = clean + fast_lysis_only 
lysis_times_adjusted = pd.read_csv("10ms_lysis_times_adjusted.csv")

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

# calculate the key time points during lysis.
fast_lysis = {}
for k in cells.keys():
    if k in clean:
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
        
        threshold_value = value_arr[peak_idx] * 0.5
        start_idx = peak_idx
        fall_idx, fall = find_crossing_point(time_arr, value_arr, threshold_value, window_length, start_idx=start_idx, mode="decreasing")
        
        fast_lysis[k] = [rise, peak, fall]

# structure the data as a table and save the result
cell_envelope_breakdown_analysis = pd.DataFrame()
cell_ids = []
rise_times = []
peak_times = []
fall_times = []
for k, v in fast_lysis.items():
    cell_ids.append(k)
    rise_times.append(v[0])
    peak_times.append(v[1])
    fall_times.append(v[2])
    
cell_envelope_breakdown_analysis["cell"] = cell_ids
cell_envelope_breakdown_analysis["rise_time"] = rise_times
cell_envelope_breakdown_analysis["peak_time"] = peak_times
cell_envelope_breakdown_analysis["fall_time"] = fall_times
        
try:
    os.mkdir("dataframes")
except:
    pass

cell_envelope_breakdown_analysis.to_csv("dataframes/cell_envelope_breakdown_analysis.csv")

