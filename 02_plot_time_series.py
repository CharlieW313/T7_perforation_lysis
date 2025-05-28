import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

### The purpose of this script is to help you quickly plot the time series data for inspection.

# plot an example plot
d = pd.read_csv("lysis_data_time_adjusted/lysis_01.csv")

plt.subplots(nrows=1, ncols=1, figsize=(12,8))
plt.plot(d["time"], d["l"], color="#420a68", label="Left of cell")
plt.plot(d["time"], d["c"], color="#932667", label="Cell")
plt.plot(d["time"], d["r"], color="#dd513a", label="Right of cell")
plt.plot(d["time"], d["st"], color="#fca50a", label="Side trench")
plt.xlabel("Time (s)", fontsize=26)
plt.ylabel("Mean intensity (AU/px)", fontsize=26)
plt.legend(fontsize=20, frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([25,50])
plt.show()
plt.close()

# make a directory to store the plots
# be wary that the try except block may mask other errors in directory creation.
try:
    os.mkdir("time_series_plots")
except:
    print("Directory already exists!")
    pass

# dictionary of cells, purpose explained in 01_time_adjust_data.py
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

# plot all data and save
for k in cells.keys():
    d = pd.read_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))
    plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    plt.plot(d["time"], d["l"], label="Left of cell, {}".format(str(k).zfill(2)))
    plt.plot(d["time"], d["c"], label="Cell, {}".format(str(k).zfill(2)))
    plt.plot(d["time"], d["r"], label="Right of cell, {}".format(str(k).zfill(2)))
    plt.plot(d["time"], d["st"], label="Side trench, {}".format(str(k).zfill(2)))
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Mean intensity (AU/px)", fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("time_series_plots/20230811_lysis_{}_full_time_series.png".format(str(k).zfill(2)), bbox_inches='tight', dpi=300)
    plt.close()