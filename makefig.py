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



indir = 'march21'
exp = 6

# t1path     =  os.path.join(indir,f"exp{exp}",'t1_mean.csv')
t2path     =  os.path.join(indir,f"exp{exp}",'t2_mean.csv')
t6path     =  os.path.join(indir,f"exp{exp}",'t6_mean.csv')
fitpath    =  os.path.join(indir,f"exp{exp}",'fit_mean.csv')
bratepath  =  os.path.join(indir,f"exp{exp}",'brate_mean.csv')



# t1       =   pd.read_csv(t1path, header=None,delimiter=' ', sep=' ').iloc[0]
t2       =   pd.read_csv(t2path, header=None,delimiter=' ', sep=' ').iloc[0]
t6       =   pd.read_csv(t6path, header=None,delimiter=' ', sep=' ').iloc[0]
fit      =   pd.read_csv(fitpath, header=None, delimiter=' ', sep=' ').iloc[0] 
brate    =   pd.read_csv(bratepath, header=None, sep=' ', delimiter=' ').iloc[0] 

time = np.arange(len(fit))

figur, axarr = plt.subplots(3)

axarr[0].plot(time, t2, label="Type 2", color="green")
axarr[0].plot(time, t6, label="Type 6", color="orange")
# axarr[0].plot(time, t2, label="Type 2", color="green")
axarr[0].set_ylabel('Individual Count')
axarr[0].legend()

axarr[1].plot(time, fit, label="Fitness")
axarr[1].set_ylabel('Populationwide Fitness')

axarr[2].plot(time, brate, label="Birthrate")
axarr[2].set_ylabel('Birthrate')

figure = plt.gcf()
figure.suptitle(f"Experimet{exp}")
figure.set_size_inches(12, 6)
figure.text(0.5, 0.04, 'BD Process Iteration(every 1k)', ha='center', va='center')
plt.show()