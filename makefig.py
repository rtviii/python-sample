from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        try:
            if not os.path.exists(string):
                os.makedirs(string, exist_ok=True)
                return string
        except:
            raise PermissionError(string)

parser = argparse.ArgumentParser(description='Simulation presets')
parser.add_argument('--indir', type=dir_path, help="""Specify the path to write the results of the simulation.""")
parser.add_argument('--t1', type=int, help="""Specify the path to write the results of the simulation.""")
parser.add_argument('--t2', type=int, help="""Specify the path to write the results of the simulation.""")
parser.add_argument('--t6', type=int, help="""Specify the path to write the results of the simulation.""")


args                          =  parser.parse_args()



indir = 'means'
exp = 12

# t1path     =  os.path.join(indir,f"exp{exp}",'t1_mean.csv')
norm6path  =  os.path.join(indir,f"exp{exp}",'norm6_mean.csv')
t6path     =  os.path.join(indir,f"exp{exp}",'t6_mean.csv')
# bratepath  =  os.path.join(indir,f"exp{exp}",'norm2_mean.csv')
# t6path     =  os.path.join(indir,f"exp{exp}",'t6_mean.csv')
# bratepath  =  os.path.join(indir,f"exp{exp}",'norm6_mean.csv')
fitpath    =  os.path.join(indir,f"exp{exp}",'fit_mean.csv')
bratepath  =  os.path.join(indir,f"exp{exp}",'brate_mean.csv')



# t1       =   pd.read_csv(t1path, header=None,delimiter=' ', sep=' ').iloc[0]
# t2       =   pd.read_csv(t2path, header=None,delimiter=' ', sep=' ').iloc[0]
t6       =   pd.read_csv(t6path, header=None,delimiter=' ', sep=' ').iloc[0]
fit      =   pd.read_csv(fitpath, header=None, delimiter=' ', sep=' ').iloc[0] 
brate    =   pd.read_csv(bratepath, header=None, sep=' ', delimiter=' ').iloc[0] 
norm6    =   pd.read_csv(norm6path, header=None, sep=' ', delimiter=' ').iloc[0] 

time = np.arange(len(fit))
figur, axarr = plt.subplots(2,2)
axarr[0,0].plot(time, t1, label="Type 1", color="green")
axarr[0,0].set_ylabel('Individual Count')
axarr[0,0].legend()

axarr[0,1].plot(time, fit, label="Fitness")
axarr[0,1].set_ylabel('Populationwide Fitness')

axarr[1,1].plot(time, brate, label="Birthrate")
axarr[1,1].set_ylabel('Birthrate')


if SHIFTING_FITNESS_PEAK:
    time2= np.arange(len(lsc[:,0]))
    axarr[1,0].plot(time2,lsc[:,0], label="Mean 1", c="cyan")
    axarr[1,0].plot(time2,lsc[:,1], label="Mean 2", c="black")
    axarr[1,0].plot(time2,lsc[:,2], label="Mean 3", c="brown")
    axarr[1,0].plot(time2,lsc[:,3], label="Mean 4", c="yellow")
    axarr[1,0].legend()


if CONNECTIVITY_FLAG:
    time2 = np.arange(len(cnt))
    axarr[1,0].plot(time2, cnt,'-', label="T1 Connectivity",c='blue')
    axarr[1,0].plot(time2, rcpt,'-', label="T1 Receptivity",c='lightblue')
    axarr[1,0].plot([],[],'*', label="(Every 100 iterations)")
    axarr[1,0].set_ylabel('Connectivity')
    axarr[1,0].legend()

figure = plt.gcf()
figure.set_size_inches(12, 6)
figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
plt.show()
