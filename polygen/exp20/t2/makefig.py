from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re


exp = 20
indir = '/home/rtviii/dev/polygenicity-simulations/polygen'
itype = 2
sim = 10


sim = int(re.search(r'\d+', sys.argv[1]).group())

countp  =  os.path.join(indir,f"exp{exp}",f't{itype}',f't{itype}_i{sim}.csv')
fitp    =  os.path.join(indir,f"exp{exp}",'fit',f'fit_i{sim}.csv')
bratep  =  os.path.join(indir,f"exp{exp}",'brate',f'brate_i{sim}.csv')
consp   =  os.path.join(indir,f"exp{exp}",f"data{sim}.parquet")

count  =  pd.read_csv(countp, header=None).loc[0,:] 
fit    =  pd.read_csv(fitp, header=None).loc[0,:] 
brate  =  pd.read_csv(bratep, header=None).loc[0,:] 
means  =  pd.read_parquet(consp)

EXTINCTION = False

if count[len(count)-1] == 1:
    EXTINCTION = True
    
mean0 = means['mean0']
mean1 = means['mean1']
mean2 = means['mean2']
mean3 = means['mean3']

time     =  np.arange(len(count))
tcolors  =  ['black','blue','green','black','black','black','pink']
time     =  np.arange(len(fit))
figur, axarr   =  plt.subplots(2,2)

axarr[0,0].plot(time, count, label="Type {}".format(itype), color=tcolors[itype])
axarr[0,0].set_ylabel('Individual Count')
if EXTINCTION:
    axarr[0,0].scatter(time[-1], 0, marker='*', s=100, c='red')
    axarr[0,0].text(time[-1]+200, 0+20, s="EXTINCTION")
axarr[0,0].legend()

axarr[0,1].plot(time, fit, label="Fitness")
axarr[0,1].set_ylabel('Populationwide Fitness')

axarr[1,1].plot(time, brate, label="Birthrate")
axarr[1,1].set_ylabel('Birthrate')

time2= np.arange(len(mean0))
axarr[1,0].plot(time2,mean0, label="Mean 1", c="cyan")
axarr[1,0].plot(time2,mean1, label="Mean 2", c="black")
axarr[1,0].plot(time2,mean2, label="Mean 3", c="brown")
axarr[1,0].plot(time2,mean3, label="Mean 4", c="yellow")
axarr[1,0].legend()

figure = plt.gcf()
figure.set_size_inches(20,8)
plt.suptitle(f"Experiment {exp}")
figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
plt.savefig(os.path.join(indir,f"exp{exp}",f'Experiment{exp}_i{sim}'),dpi=100)
print(f"Saved experiment {sim}")
# plt.show()
