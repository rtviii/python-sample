import matplotlib.pyplot as plt
import numpy as np





plt.style.use('dark_background')
time = np.arange(len(fitness))
figur, axarr = plt.subplots(3)
axarr[0].plot(time, t1, label="Population Size", c='white')
axarr[0].set_ylabel('Individual Count')

axarr[1].plot(time, fitness, label="Fitness", c='orange')
axarr[1].set_ylabel('Populationwide Fitness')

axarr[2].plot(time, brate, label="Birthrate",c='blue')
axarr[2].set_ylabel('Birthrate')
figure = plt.gcf()
figure.set_size_inches(12, 6)
figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
plt.show()