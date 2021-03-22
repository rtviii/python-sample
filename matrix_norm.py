import numpy as np
import math 


t = np.array([
[-0.1,0.24,-0.59],
[0.5,0,0.5],
[0.9,0,0.2]
])


ind1 = np.array(
    [ 

    [1,0],
    [0,1] 

    ], dtype=np.float64
)
ind6 = np.array([
                            [1,0],
                            [1,0],
                            [0,1],
                            [0,1],
                            ],
                            np.float64)

s=0
# arr=[ind1]*3
arr2=[ind6]*3
# for x in arr:
#     s+=np.linalg.norm(x,ord='fro')
for x in arr2:
    s+=np.linalg.norm(x,ord='fro')
print(s/3)