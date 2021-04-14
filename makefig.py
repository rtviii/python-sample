from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob, os

# datafolder  =  sys.argv[1]
exp         =  int( sys.argv[1] )


mean0all  =  0
mean1all  =  0
mean2all  =  0
mean3all  =  0
fitall    =  0
brateall  =  0
countall  =  0


c = 0

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/rt/exp{exp}/*.parquet"):
    print("FILE", file)
    data       =  pd.read_parquet(file)
    # print(data)

    EXTINCTION = False
    if np.array(data[f't{exp}'])[len(np.array(data[f't{exp}']))-1] == 1:
        EXTINCTION = True
        print("Found exticnt {}".format(exp))
        # l =len(np.array(data[f't{exp}']))-1
        # print(l)
        continue


    mean0all  +=  np.array(data['mean0'])
    mean1all   +=  np.array( data['mean1'] )
    mean2all   +=  np.array( data['mean2'] )
    mean3all   +=  np.array( data['mean3'])


    fitall     +=  np.array( data['fit'] )
    brateall   +=  np.array( data['brate'] )
    countall   +=  np.array( data[f't{exp}'] )
    c+=1





mean0all  =  mean0all/c
mean1all  =  mean1all/c
mean2all  =  mean2all/c
mean3all  =  mean3all/c
fitall    =  fitall  /c
brateall  =  brateall/c
countall  =  countall/c


time    =  np.arange(len(countall))
time    =  np.arange(len(fitall))
figur, axarr  =  plt.subplots(2,2)

axarr[0,0].plot(time, countall, label="Type {}".format(exp), color="blue")
axarr[0,0].set_ylabel('Individual Count')
# if EXTINCTION:
#     axarr[0,0].scatter(time[-1], 0, marker='*', s=100, c='red')
#     axarr[0,0].text(time[-1]+200, 0+20, s="EXTINCTION")
axarr[0,0].legend()

axarr[0,1].plot(time, fitall, label="Fitness")
axarr[0,1].set_ylabel('Populationwide Fitness')

axarr[1,1].plot(time, brateall, label="Birthrate")
axarr[1,1].set_ylabel('Birthrate')

time2= np.arange(len(mean0all))
axarr[1,0].plot(time2,mean0all, label="Mean 1", c="cyan")
axarr[1,0].plot(time2,mean1all, label="Mean 2", c="black")
axarr[1,0].plot(time2,mean2all, label="Mean 3", c="brown")
axarr[1,0].plot(time2,mean3all, label="Mean 4", c="yellow")
axarr[1,0].legend()

figure = plt.gcf()
figure.set_size_inches(20,8)
plt.suptitle(f"Experiment {exp}")
figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
plt.show()
