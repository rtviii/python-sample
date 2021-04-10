from __future__ import annotations
from functools import reduce 
import functools
from operator import xor
from pprint import pprint
import xxhash
import csv
import sys, os
import numpy as np
from typing import  Callable, List, Tuple
import math
import argparse
import pandas as pd


x = np.array([
                        [1,0,0,0],
                        [1,1,1,0],
                        [0,1,1,0],
                        [0,0,0,1],
                    ], 
                    dtype=np.float64
                    )

mask = np.array([1,1,-1,1])
y2 = x * mask[:,None]  * mask

t = np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64) * np.array([-1,1,-1,1]) * np.array([-1,1,-1,1])[:,None]