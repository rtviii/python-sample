import matplotlib.pyplot as plt
import os, sys
import numpy as np
import pandas as pd





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