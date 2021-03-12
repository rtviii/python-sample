import pandas as pd
import time 
import matplotlib.pyplot as plt
import dask.dataframe as dd
from functools import reduce
from sys import getsizeof
import numpy as np
from dask.distributed import Client


client = Client(n_workers=3, threads_per_worker=4, processes=False, memory_limit='2GB')


dfname = f'batch3_*.csv'

t1 =time.time()
df = dd.read_csv(dfname, usecols=[0,1,2,3,4,5,6,7,8,9], sample=6000000, header=None, dtype=np.int16)

print(np.sum(list(df[0]))/100)
print(np.sum(list(df[1]))/100)
print(np.sum(list(df[2]))/100)

t2 =time.time()

print("\nWalltime: ",t2-t1,"\n")