import matplotlib.pyplot as plt
import numpy as np
import sys,os
import pandas as pd







# simn =sys.argv[1]

# simdata = pd.read_csv(f"t3_vs_t5_{simn}.csv_(itern=100000).csv",header=None)

# type3    =  simdata.iloc[0,:]
# type5    =  simdata.iloc[1,:]

# time = np.arange(1e5)

# plt.plot(time, type3, label="Type3")
# plt.plot(time, type5, label="Type5")
# plt.legend()
# plt.ylabel("Individual Count")
# plt.xlabel("BD Process Iter")
# figure = plt.gcf()
# figure.set_size_inches(12, 6)
# plt.savefig(f"sim_1e5_{simn}.png", dpi=1200)

# !--------------MEAN
t3_alldfs = pd.DataFrame()
t5_alldfs = pd.DataFrame()

for x in range(1,8):
    simdata = pd.read_csv(f"t3_vs_t5_{x}.csv_(itern=100000).csv",header=None)
    type3    =  simdata.iloc[0,:]
    type5    =  simdata.iloc[1,:]
    t3_alldfs[x] = type3
    t5_alldfs[x] = type5

avgt3= t3_alldfs.sum(axis=1)/10
avgt5= t5_alldfs.sum(axis=1)/10
time = np.arange(len(avgt3))

plt.plot(time, avgt3, label="Type3")
plt.plot(time, avgt5, label="Type5")
plt.legend()
plt.ylabel("Individual Count")
plt.xlabel("BD Process Iter")
plt.title(label="Mean")
figure = plt.gcf()
figure.set_size_inches(12, 6)
plt.savefig(f"sim_mean_1e5.png", dpi=1200)