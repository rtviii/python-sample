import matplotlib.pyplot as plt
import os, sys
import numpy as np
import pandas as pd
from __future__ import annotations
from functools import reduce
import time
from operator import xor
import csv
import sys, os
import numpy as np
from typing import  Callable, List
import math
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
parser.add_argument('--outdir', type=dir_path, help="""Specify the path to write the results of the simulation.""")
parser.add_argument("-it", "--itern", type=int, help="The number of iterations")
parser.add_argument("-sim", "--siminst", type=int, help="Simulation tag for the current instance.")
parser.add_argument("-SFL", "--shifting_landscape", type=int, choices=[0,1], help="Flag for whether the fitness landscape changes or not.")
parser.add_argument("-V", "--verbose", type=int, choices=[0,1])
parser.add_argument("-plt", "--toplot", type=int, choices=[0,1], help="Flag for whether the fitness landscape changes or not.")

args                          =  parser.parse_args()
itern                         =  int(args.itern)
instance                      =  int(args.siminst)
shifting_landscape_flag       =  bool(args.shifting_landscape)
toplot                        =  bool(args.toplot)
outdir                        =  str(args.outdir)
verbose                       =  bool(args.verbose)

MUTATION_RATE_ALLELE          =  0.001
MUTATION_VARIANTS_ALLELE      =  np.arange(-1,1,0.01)
MUTATION_RATE_DUPLICATION     =  0
MUTATION_RATE_CONTRIB_CHANGE  =  0
DEGREE                        =  1




indir = 'march20'

t1path     =  os.path.join(indir,'t1','t1_i1.csv')
fitpath    =  os.path.join(indir,'avg_fitness','avg_fitness_i1.csv')
bratepath  =  os.path.join(indir,'brate','brate_i1.csv')



t1       =  pd.read_csv(t1path, header=None).iloc[0]
fitness  =  pd.read_csv(fitpath, header=None).iloc[0]
brate    =  pd.read_csv(bratepath, header=None).iloc[0]

time = np.arange(len(fitness))



figur, axarr = plt.subplots(3)
axarr[0].plot(time, t1, label="Population Size")
axarr[0].set_ylabel('Individual Count')

axarr[1].plot(time, fitness, label="Fitness")
axarr[1].set_ylabel('Populationwide Fitness')

axarr[2].plot(time, brate, label="Birthrate")
axarr[2].set_ylabel('Birthrate')
figure = plt.gcf()
figure.set_size_inches(12, 6)
figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
plt.show()