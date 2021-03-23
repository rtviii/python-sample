import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt



t1     =  pd.DataFrame()
fit    =  pd.DataFrame()
brate  =  pd.DataFrame()
cnt    =  pd.DataFrame()
rcpt   =  pd.DataFrame()

for _,i in  enumerate(glob.glob('./connectivity*')):
    df   =  pd.read_parquet(i)
    cnt  [_]  =  df['cnt']
    rcpt [_]  =  df['rcpt']

cnt   =  cnt.mean(axis=1)
rcpt  =  rcpt.mean(axis=1)


for _,i in  enumerate(glob.glob('./data*')):
    df  =  pd.read_parquet(i)
    fit  [_]  =  df['fit']
    t1   [_]  =  df['t1']
    brate[_]  =  df['brate']

t1     =  t1.mean(axis=1)
brate  =  brate.mean(axis=1)
fit    =  fit.mean(axis=1)


exp = 7





# print( con)
time = np.arange(len(fit))
time2 = np.arange(len(rcpt))

figur, axarr = plt.subplots(2,2)

axarr[0,0].plot(time, t1, label="Type 1", color="blue")
axarr[0,0].set_ylabel('Individual Count')
axarr[0,0].legend()

axarr[0,1].plot(time, fit, label="Fitness")
axarr[0,1].set_ylabel('Populationwide Fitness')

axarr[1,1,].plot(time, brate, label="Birthrate")
axarr[1,1].set_ylabel('Birthrate')

axarr[1,0].plot(time2, cnt, label="Connectivity (Avg. per gene)")
axarr[1,0].plot(time2, rcpt, label="Receptivity (Avg. per trait)")
axarr[1,0].plot([], [],'*', label="Every 10k iterations")
axarr[1,0].legend()
axarr[1,0].set_ylabel('Connectivity')

figure = plt.gcf()
figure.suptitle(f"Experimet{exp}")
figure.set_size_inches(12, 6)
figure.text(0.5, 0.04, 'BD Process Iteration(every 1k)', ha='center', va='center')
plt.show()