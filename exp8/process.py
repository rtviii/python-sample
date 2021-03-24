import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt



t2     =  pd.DataFrame()
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
    if _ < 15:
        df  =  pd.read_parquet(i)
        fit  [_]  =  df['fit']
        t2   [_]  =  df['t2']
        brate[_]  =  df['brate']
    else:
        break

t2     =  t2.mean(axis=1)
brate  =  brate.mean(axis=1)
fit    =  fit.mean(axis=1)


exp = 8





# print( con)
time = np.arange(len(fit))
time2 = np.arange(len(rcpt))

figur, axarr = plt.subplots(2,2)

axarr[0,0].plot(time, t2, label="Type 2", color="blue")
axarr[0,0].set_ylabel('Individual Count')
axarr[0,0].legend()

axarr[0,1].plot(time, fit, label="Fitness", color='blue')
axarr[0,1].set_ylabel('Populationwide Fitness')

axarr[1,1,].plot(time, brate, label="Birthrate", color='blue')
axarr[1,1].set_ylabel('Birthrate')

axarr[1,0].plot(time2, cnt, label="Connectivity(per-gene)", color='grey')
axarr[1,0].tick_params(axis='y', labelcolor="grey")
ax2 = axarr[1,0].twinx()
ax2.plot(time2, rcpt, label="Receptivity(per-trait)", color='orange')
ax2.tick_params(axis='y', labelcolor="orange")
axarr[1,0].legend()
ax2.legend()


figure = plt.gcf()
figure.suptitle(f"Experimet{exp}")
figure.set_size_inches(12, 6)
figure.text(0.5, 0.04, 'BD Process Iteration(every 1k)', ha='center', va='center')
plt.show()