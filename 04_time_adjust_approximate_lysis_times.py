import pandas as pd
import numpy as np
import os

### The table '10ms_lysis_times.csv' contains approximate lysis times obtained by inspecting the image data. However, this table estimates the lysis time by multiplying the time point by 10 ms.
### Since the true imaging interval for each trench is 1.3% to 2.0% larger than 10 ms, this script corrects for this small error in the estimated time. 
### These estimate times will later inform algorithms for quantitatively finding the start of perforation and lysis. 

lysis_times = pd.read_csv("10ms_lysis_times.csv")

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

# adjust timings in each trench by the temporal frame spacing to correct for the 1.3 to 2.0% error in frame spacing on 10 ms
# temporal frame spacings found also in image metadata (along with coefficient of variation).
trench_1_fs = 0.010192040380847209 # frame spacing in seconds
trench_2_fs = 0.010131019743528872
trench_3_fs = 0.01018282972445327
trench_4_fs = 0.01020342320864684
trench_5_fs = 0.010183534963434706
frame_spacing = [trench_1_fs, trench_2_fs, trench_3_fs, trench_4_fs, trench_5_fs]

# adjust the lysis times
lysis_times_adjusted = lysis_times.copy()
lysis_times_adjusted = lysis_times_adjusted[lysis_times_adjusted["cell"].isin(cells.keys())]
adjusted = []
for cell, v in cells.items():
    fs = frame_spacing[v[0]-1]
    l = lysis_times[lysis_times["cell"] == cell]
    t = l["lysis_t_start"].tolist()[0]
    adjusted.append(t * fs)

# save the result
lysis_times_adjusted["lysis_t"] = adjusted
lysis_times_adjusted.to_csv("10ms_lysis_times_adjusted.csv")