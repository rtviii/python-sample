from __future__ import annotations
from functools import reduce 
import functools
from operator import xor
import xxhash
import csv
import sys, os
import numpy as np
from typing import  Callable, List, Tuple
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

VERBOSE = False


def dir_path(string):
    if string == "0":
        return None
    if os.path.isdir(string):
        return string
    else:
        try:
            if not os.path.exists(string):
                os.makedirs(string, exist_ok=True)
                return string
        except:
            raise PermissionError(string)

parser = argparse.ArgumentParser(description='Simulation presets')
parser.add_argument('-save',    '--outdir',              type=dir_path, help="""Specify the path to write the results of the simulation.""")
parser.add_argument("-it",      "--itern",               type=int,      help="The number of iterations")
parser.add_argument("-sim",     "--siminst",             type=int,      help="Simulation tag for the current instance.")
parser.add_argument("-SP",      "--shifting_peak",       type=int,      choices=[-1,0,1], help="Flag for whether the fitness landscape changes or not.")
parser.add_argument("-plot",    "--toplot",              type=int,      choices=[0,1])
parser.add_argument("-con",     "--connectivity",        type=int,      choices=[0,1])
parser.add_argument("-exp",     "--experiment",          type=int)
parser.add_argument('-t',       '--type',                type=int,      required=True, help='Types involved in experiment')
parser.add_argument("-V",       "--verbose",             type=int,      choices=[0,1])


args      =  parser.parse_args()
itern     =  int(args.itern if args.itern is not None else 0)
instance  =  int(args.siminst if args.siminst is not None else 0)
toplot    =  bool(args.toplot if args.toplot is not None else 0)
outdir    =  args.outdir if args.toplot is not None else 0

INDTYPE                       =  args.type
EXPERIMENT                    =  args.experiment if args.experiment is not None else "Unspecified"
VERBOSE                       =  True if args.verbose is not None and args.verbose !=  0 else False
SHIFTING_FITNESS_PEAK         =  args.shifting_peak if args.shifting_peak is not None else False
CONNECTIVITY_FLAG             =  args.connectivity if args.connectivity is not None else False
CON_SPARSE                    =  1000
MUTATION_RATE_ALLELE          =  0.001
MUTATION_VARIANTS_ALLELE      =  np.arange(-1,1,0.01)
MUTATION_RATE_DUPLICATION     =  0
MUTATION_RATE_CONTRIB_CHANGE  =  0
DEGREE                        =  1
BRATE_DENOM                   =  0.0008
COUNTER_RESET                 =  1024
STD                           =  1
AMPLITUDE                     =  1
LANDSCAPE_INCREMENT           =  0.5
INDIVIDUAL_INITS     =  {   

   "1":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ], dtype=np.float64)
   },
   "2":{
       'trait_n':4,
        'alleles'       :  np.array([1,0,0,1,0,0,1,0,0,1,0,0], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,0,0,0,0,0,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,1,1,1],
                    ],dtype=np.float64)
   },
   "6":{
       "trait_n"          :  4,
        'alleles'       :  np.array([1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                            [1,0],
                            [1,0],
                            [0,1],
                            [0,1],
                            ],dtype=np.float64)}
}



class Fitmap():
    def __init__(self,std:float, amplitude:float, mean):
        self.std        :  float = std
        self.amplitude  :  float = amplitude
        self.mean          = mean
    def getMap(self):
        def _(phenotype:np.ndarray):
            return self.amplitude*math.exp(-(np.sum(((phenotype - self.mean)**2)/(2*self.std**2))))
        return _

class GPMap():
    def __init__(self,contributions:np.ndarray) -> None:
        self.coeffs_mat     =  contributions

    # @functools.lru_cache
    def map_phenotype(self, alleles:np.ndarray  )->np.ndarray:
        #? The whole purpose of gpmap is to define a map from genotype to phenotype
        #? Here phenotype is calculated
        """Phenotype is calculated:
        degrees are applied
        weights are applied
        column-wise sum
        """

        return  np.sum(self.coeffs_mat * ( alleles ** 1), axis=1)

class Individual:
    def __init__(self, alleles:np.ndarray, ind_type:int):
        self.ind_type   =  ind_type
        self.alleles    =  alleles
        self.phenotype  =  []

    def give_birth(self)->Individual:

        def mutation_allele():
            return np.random.choice(MUTATION_VARIANTS_ALLELE)

        # template alleles from parent
        alleles_copy  =  np.copy(self.alleles)

        #whether template alleles suffer a mutation; otherwise dont cbother copying anything
        did_mutate = False
        for index, gene in enumerate( self.alleles.tolist() ):
            if np.random.uniform() <= MUTATION_RATE_ALLELE:
                did_mutate = True
                alleles_copy[index] = mutation_allele()

        if did_mutate:
            nascent = Individual(alleles_copy, self.ind_type)
        else:
            # if haven't mutated -- gpmap is remains the same as parent
            nascent = Individual(self.alleles, self.ind_type)
        return nascent

class Universe:
    def __init__(self, initial_population:List[Individual], GPMap:GPMap,Fitmap:Fitmap) -> None:
        self.population   =  []
        self.GPMap        =  GPMap
        self.Fitmap       =  Fitmap
        self.phenotypeHM  =  {}
        self.poplen       =  0
        self.iter         =  0
        self.avg_fitness  =  0

        self.drate = 0
        self.brate = 0

        for i in initial_population:
            self.birth(i)
    
        self.avg_fitness  =  self.get_avg_fitness()
        self.brate        =  ( self.avg_fitness )/( self.avg_fitness + self.poplen * BRATE_DENOM)
        self.drate        =  1 - self.brate

    def _hashalls(self,_:Individual)->str:
        return xxhash.xxh64(np.array2string(_.alleles)).hexdigest()

    def get_avg_fitness(self):
        """From the genotype hashmap::: collect each genotype's fitness multiplying by the number of individuals 
        present with such."""
        return reduce(lambda x,y: x + y['f']*y['n'] , self.phenotypeHM.values(),0)/self.poplen
    def tick(self, V:bool=False)->None:
        
        self.iter  +=1
        self.avg_fitness     =  self.get_avg_fitness()
        self.brate  =  ( self.avg_fitness )/( self.avg_fitness + self.poplen * BRATE_DENOM)
        self.drate  =  1 - self.brate

        pick =  np.random.choice([1, -1], p=[self.brate, self.drate])

        if pick > 0:
            chosen:Individual =  np.random.choice(self.population, replace=False)
            u.birth(chosen.give_birth())
        else:
            chosen = np.random.choice(self.population)
            u.death(chosen)

        
    
    def death(self,_:Individual):
        self.phenotypeHM[self._hashalls(_)]['n']-=1
        if self.phenotypeHM[self._hashalls(_)]['n'] == 0:
            self.phenotypeHM.pop(self._hashalls(_))
        self.population.remove(_)
        self.poplen-=1

    def birth(self,individ:Individual)->None:

        K = self._hashalls(individ)

        if K in self.phenotypeHM:
            #*Genotype is present
            self.population.append(individ)
            self.phenotypeHM [K]['n']+=1
            self.poplen +=1

        else:
            fitval = self.get_fitness(individ)
            self.population.append(individ)
            self.phenotypeHM[K] = {
                'a':individ.alleles,
                'f':fitval,
                'n':1
            }
            self.poplen+=1
            
    def get_fitness(self,ind:Individual) -> float:
        K                 =  xxhash.xxh64(np.array2string(ind.alleles)).hexdigest()

        if K in self.phenotypeHM:
            return self.phenotypeHM[K]['f']
        else:
            phenotype         =  self.GPMap.map_phenotype(ind.alleles)
            fitval            =  self.Fitmap.getMap()(phenotype)
            return fitval


count              =  []
fit                =  []
brate              =  []
if SHIFTING_FITNESS_PEAK:
    lsc  =  np.array([], ndmin=2)
if CONNECTIVITY_FLAG:
    cnt   =  []
    rcpt  =  []


mean         =  np.array([0.0,0.0,0.0,0.0], dtype=np.float64)
ASYM_SWITCH  =  False
EXTINCTION   =  False




gpm1  =  GPMap(INDIVIDUAL_INITS['1']['coefficients'])
ftm   =  Fitmap(1,1,[0,0,0,0])
ind1  =  Individual(INDIVIDUAL_INITS['1']['alleles'], 1)
u = Universe([ind1]*400,gpm1,ftm)
u.tick(V=VERBOSE)


for it in range(itern):

    if u.poplen == 0:
        EXTINCTION = True
        break

    count.append(u.poplen)
    fit.append(u.avg_fitness)
    brate.append(u.brate)


    if SHIFTING_FITNESS_PEAK:
        lsc= np.append(lsc, mean)

    if (not (it + 1 )  & (COUNTER_RESET -1 ) ) and SHIFTING_FITNESS_PEAK:        #! Correlated
        if SHIFTING_FITNESS_PEAK == 1:
            if np.max(mean) > 0.9:
                LANDSCAPE_INCREMENT    =  -0.5
                mean += LANDSCAPE_INCREMENT
            elif np.max(mean) < -0.9:
                LANDSCAPE_INCREMENT    =  0.5
                mean += LANDSCAPE_INCREMENT
            else:
                coin      = np.random.choice([-1,1 ])
                mean[0:] += coin*LANDSCAPE_INCREMENT
            
        else:
            for i,x in enumerate(mean):
                if abs(x) == 1:
                    mean[i] += -mean[i]/2
                else:
                    mean[i] += np.random.choice([0.5,-0.5])

        u.Fitmap.mean=mean
    u.tick()


if SHIFTING_FITNESS_PEAK:
    lsc = np.reshape(lsc, (-1,4))

[count,fit,brate]=[*map(lambda x: np.around(x,5), [count,fit,brate])]

data = pd.DataFrame({
    f"t{INDTYPE}"  :  count,
      "fit"        :  fit,
      "brate"      :  brate,
})


if toplot:
    tcolors = ['black','blue','green','black','black','black','pink']
    time = np.arange(len(fit))
    figur, axarr = plt.subplots(2,2)
    axarr[0,0].plot(time, count, label="Type {}".format(INDTYPE), color=tcolors[INDTYPE])
    axarr[0,0].set_ylabel('Individual Count')
    axarr[0,1].plot(time, fit, label="Fitness")
    axarr[0,1].set_ylabel('Populationwide Fitness')
    axarr[1,1].plot(time, brate, label="Birthrate")
    axarr[1,1].set_ylabel('Birthrate')


    if SHIFTING_FITNESS_PEAK:
        time2= np.arange(len(lsc[:,0]))
        axarr[1,0].plot(time2,lsc[:,0], label="Mean 1", c="cyan")
        axarr[1,0].plot(time2,lsc[:,1], label="Mean 2", c="black")
        axarr[1,0].plot(time2,lsc[:,2], label="Mean 3", c="brown")
        axarr[1,0].plot(time2,lsc[:,3], label="Mean 4", c="yellow")
        axarr[1,0].plot([],[], label="Landscape {}".format("Correlated" if SHIFTING_FITNESS_PEAK>0 else "Uncorrelated"),c="black")
        axarr[1,0].legend()

    if CONNECTIVITY_FLAG:
        time2 = np.arange(len(cnt))
        axarr[1,0].plot(time2, cnt,'-', label="T1 Connectivity",c='blue')
        axarr[1,0].plot(time2, rcpt,'-', label="T1 Receptivity",c='lightblue')
        axarr[1,0].plot([],[],'*', label="(Every 100 iterations)")
        axarr[1,0].set_ylabel('Connectivity')
        axarr[1,0].legend()

    if EXTINCTION:
        axarr[0,0].scatter(time[-1], 0, marker='H', s=50)
        axarr[0,0].text(time[-1]+0.15, 0+0.15, s="EXTINCTION")

    figure = plt.gcf()
    figure.suptitle("Experiment {}".format(EXPERIMENT))
    figure.set_size_inches(12, 6)
    figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
    plt.show()


from pprint import pprint
pprint(u.phenotypeHM)