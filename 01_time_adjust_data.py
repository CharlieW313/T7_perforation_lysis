import pandas as pd
import numpy as np
import os

### The aim of this script is to collate the individual data files for each cell, 
### and to adjust the time between frames to be consistent with the 
### experimentally recorded imaging interval (see image metadata .txt files).
### Note that the times start at zero independently for each trench.

# Create an index to help load in the intensity data.
# As the masks are static and the cells sometimes move in the time period leading up to lysis,
# the start_timepoint is adjusted to be at a suitable start time for the analysis.
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

# make a directory to store the image data
# be wary that the try except block may mask other errors in directory creation.
try:
    os.mkdir("lysis_data_time_adjusted")
except:
    print("Directory already exists!")
    pass

# save the time adjusted lysis data
for k, v in cells.items():
    d = pd.DataFrame()
    dl = pd.read_csv("lys_{}_l.csv".format(str(k).zfill(2)))
    dc = pd.read_csv("lys_{}_c.csv".format(str(k).zfill(2)))
    dr = pd.read_csv("lys_{}_r.csv".format(str(k).zfill(2)))
    dt = pd.read_csv("lys_{}_st.csv".format(str(k).zfill(2)))
    
    d["timepoint"] = dl["Slice"].copy() - 1
    d["time"] = (d["timepoint"] + v[1]) * frame_spacing[v[0] - 1]
    d["cell"] = k
    d["trench"] = v[0]
    
    d["l"] = dl["Mean"].copy()
    d["c"] = dc["Mean"].copy()
    d["r"] = dr["Mean"].copy()
    d["st"] = dt["Mean"].copy()
    
    d.to_csv("lysis_data_time_adjusted/lysis_{}.csv".format(str(k).zfill(2)))