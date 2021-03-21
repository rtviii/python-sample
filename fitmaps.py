import numpy as np
import math

def FITMAP(x,std:float=1, height:float=1, peak:float=0):
    return height*math.exp(-(sum(((x - peak)**2)/(2*std**2))))

def multi_fitmap(x,std=[1,1,1], height=1, peak=[0,0,0],trait_n:int=2):
    S=0

    for i,v in enumerate(x):
        S += ( ( v - peak[i])**2 )/(2*std[i]**2)

    return height * math.exp(-S)
    

print(FITMAP(np.array([1,1])))
print(multi_fitmap([1,1]))

