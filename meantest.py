import numpy as np
import math 


def fit(phenotype, nean, std):

    std        = 0
    amplitude  = 1
    mean       = nean
    return amplitude*math.exp(
        -(np.sum(
            ((phenotype - mean)**2)
            /
            (2*std**2)))
            )


s = 1
p2 = np.array([-0.2, 0.1])
p1 = np.array([1, 1])


m = 1
print("mean :{}\t| FIT(phenotype[1,1]) :{}\t| FIT(phenotype[-0.2,0.1]):{}|  ".format(m,fit(p1,m,s), fit(p2,m,s)))
m = 2
print("mean :{}\t| FIT(phenotype[1,1]) :{}\t| FIT(phenotype[-0.2,0.1]):{}|  ".format(m,fit(p1,m,s), fit(p2,m,s)))
m = 2.5