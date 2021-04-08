
import sys
from numpy.random import choice
import xxhash
import numpy as np

def _hashalls(_)->str:
    return xxhash.xxh64(np.array2string(_)).hexdigest()

x = set([])


for i in np.linspace(-1,1,200):
    for j in np.linspace(-1,1,200):
        for r in np.linspace(-1,1,200):
            for w in np.linspace(-1,1,200):
                val = np.round([i,j,r,w], 2)
                h = _hashalls(val)
                if h in x:
                    print("COLLISION HAS OCCURRED: ", val)
                    break
                x.add(h)

print("No collisions. ")









