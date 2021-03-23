import numpy as np
import math


phen = np.array([1,1,1,1])


def FITMAP(x,std:float=1, height:float=1, peak=np.array([0,0,0,0])):
    return height*math.exp(-(sum(((x - peak)**2)/(2*std**2))))


print(FITMAP(phen))
    
