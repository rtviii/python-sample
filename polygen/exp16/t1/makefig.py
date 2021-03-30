from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re


exp = 16
indir = '/home/rtviii/dev/polygenicity-simulations/polygen/'
itype = 1




time     =  np.arange(2.5e6)
tcolors  =  ['black','blue','green','black','black','black','pink']
for sim in os.listdir(os.path.join(indir,f'exp{exp}',f't{itype}')):
    sim     =  int(re.search(r'\d+', sim[4:]).group()) 
    if type(sim) != int:
        continue
    print(sim)
    # countp  =  os.path.join(indir,f"exp{exp}",f't{itype}',f't{itype}_i{sim}.csv')
    # count   =  pd.read_csv(countp, header=None).loc[0,:]
    # plt.plot(np.arange(len(count)), count, color=tcolors[itype])


# figure = plt.gcf()
# figure.set_size_inches(20,8)
# plt.suptitle(f"Experiment {exp} | Type 1")
# figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
# plt.savefig(os.path.join(indir,f"exp{exp}",f'Experiment{exp}_all.png'),dpi=100)
# plt.show()
