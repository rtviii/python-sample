import matplotlib.pyplot as plt
import numpy as np
import sys,os
import pandas as pd







simn =sys.argv[1]

simdata = pd.read_csv(f"t3_vs_t5_{simn}.csv_(itern=100000).csv",header=None)


time = np.arange(1e5)

plt.plot(time, type3, label="Type3")
plt.plot(time, type5, label="Type5")
plt.legend()
plt.ylabel("Individual Count")
plt.xlabel("BD Process Iter")
figure = plt.gcf()
figure.set_size_inches(12, 6)
plt.savefig(f"sim_1e5_{simn}.png", dpi=1200)