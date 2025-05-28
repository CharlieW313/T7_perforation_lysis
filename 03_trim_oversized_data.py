import pandas as pd
import numpy as np
import os

### Most time series phase contrast intensity time series were 10000 time points long. Running the Fiji intensity profile function took excessive amounts of time for more time points.
### It was easiest to load in stacks of data starting at a multiple of 10000, e.g. loading the time points 10000 to 19999 using the regular expression '(xy000_PC_T1.....png)' in
### the filter box while using the Fiji 'Import image sequence' function. However, when the lysis event crossed over a multiple of 10000, a longer image sequence was used to capture the important features.
### This script trims the oversized data down to the standard size of 10000 time points, so that there is less irrelevant data which could potentially produce errors in the code if not removed.
### The table '10ms_lysis_fiji_data_summary.csv' summarises the original time point ranges. The reason for the trim chosen in each case can be inferred from the time series plots (02_plot_time_series.py).

# {cell: new_timepoint_range}
trim = {24: [50, 150],
        30: [150, 250],
        42: [250, 350],
        44: [350, 450],
        45: [630, 730],
        46: [630, 730],
        47: [650, 750],
        48: [650, 750]}

# trim and update the table
for k, v in trim.items():
    d = pd.read_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))
    d = d[(d["time"] >= v[0]) & (d["time"] < v[1])]
    d.to_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))
    
