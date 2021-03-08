import pandas as pd
import time 
import matplotlib.pyplot as plt
import dask.dataframe as dd
from functools import reduce
from dask.distributed import Client

client = Client(n_workers=3, threads_per_worker=4, processes=False, memory_limit='2GB')


dfname = f'batch3_*.csv'

t1 =time.time()
df = dd.read_csv(dfname, usecols=[0,1,2,3,4,5,6,7,8,9], sample=6000000, header=None)
print(reduce(lambda x,y: x+y, list(df[0]))/10)

t2 =time.time()

print("\nWalltime: ",t2-t1,"\n")