from __future__ import annotations
from functools import reduce
import time
from operator import xor
import csv
import sys, os
import numpy as np
from typing import  Callable, List
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def dir_path(string):
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
parser.add_argument('--outdir', type=dir_path, help="""Specify the path to write the results of the simulation.""")
parser.add_argument("-it", "--itern", type=int, help="The number of iterations")
parser.add_argument("-sim", "--siminst", type=int, help="Simulation tag for the current instance.")
parser.add_argument("-SFL", "--shifting_landscape", type=int, choices=[0,1], help="Flag for whether the fitness landscape changes or not.")
parser.add_argument("-plot", "--toplot", type=int, choices=[0,1])


args                     =  parser.parse_args()
itern                    =  int(args.itern)
instance                 =  int(args.siminst)
toplot                   =  bool(args.toplot)
shifting_landscape_flag  =  bool(args.shifting_landscape)
outdir                   =  str(args.outdir)


MUTATION_RATE_ALLELE          =  0.01
MUTATION_VARIANTS_ALLELE      =  np.arange(-1,1,0.01)
MUTATION_RATE_DUPLICATION     =  0.00
MUTATION_RATE_CONTRIB_CHANGE  =  0.00
DEGREE                        =  1
resetcounterat                =  1024 * 2
BRATE_DENOM                   =  0.001

SHIFTING_FITNESS_PEAK         =  False or shifting_landscape_flag

INDIVIDUAL_INITS              =  {   
    "1.1":{
        'trait_n' :3,
        'alleles'       :  np.array([1.0, 1.0, 0], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0],
                        [0,1,0],
                        [0,0,1],
                    ], dtype=np.float64)
   },
   "1.2":{
        'trait_n' :2,
        'alleles'       :  np.array([1.0,1.0], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1.0,0.0],
                        [0.0,1.0]]
                        , dtype=np.float64)
   },
   "1.4":{
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
                            ],dtype=np.float64)
   }
}


class Fitmap():

    def __init__(self,std:float, amplitude:float, mean:float):
        self.std        :  float = std
        self.amplitude  :  float = amplitude
        self.mean       :  float = mean

    def getMap(self):
        def _(phenotype):
            return self.amplitude*math.exp(-(np.sum(((phenotype - self.mean)**2)/(2*self.std**2))))
        return _

class GPMap():

    def __init__(self,alleles:np.ndarray, trait_n:int, deg:int) -> None:
        self.trait_n        =  trait_n
        self.deg            =  deg

        self.alleles        =  alleles
        self.genome_length  =  len(alleles)

        # Coefficients and degree matrices that constitute the gp_map
        self.coeffs_mat   =  None 
        self.degrees_mat  =  None
        
        self.deg_init()     # Initialized to defaults when the object is created
        self.coef_init()    # Initialized to defaults when the object is created

    def coef_init(self, custom_coeffs=None):

        if custom_coeffs is not None: 
            self.coeffs_mat = custom_coeffs
            return

        coeffs = np.full((self.trait_n, self.genome_length), fill_value=0)
        for x in range(self.trait_n):
            if x < self.genome_length:
                coeffs[x][x] = 1
        self.coeffs_mat = coeffs

        return 

    def deg_init(self, custom_degrees=None):

        if custom_degrees is not None: 
            self.degrees_mat = custom_degrees
            return

        self.degrees_mat = np.full((self.trait_n,self.genome_length), fill_value=1)
        return 


    def _genesPerTrait(self)->float:
        for trait in self.coeffs_mat:
            for gene in trait:
                print(gene)
        return 0
    def _contribsPerGene(self)->float:
        return 0

    def map_phenotype(self)->List[float]:


        #? The whole purpose of gpmap is to define a map from genotype to phenotype
        #? Here phenotype is calculated
        """Phenotype is calculated:
        degrees are applied
        weights are applied
        column-wise sum
        """
        
        return  np.sum(self.coeffs_mat * ( self.alleles ** self.degrees_mat), axis=1)

class Population:

    """A population class"""
    def __init__(self, fitmap:Fitmap, initial_population=[]):

        self.population:List[Individ_T]     = []
        self.typecount_dict                 = {}
        self.poplen: int                    = 0
        self.average_fitness:float          = 0
        self.brate: float                   = 0
        self.drate: float                   = 0
        self.fitmap                         = fitmap

        if len(initial_population) != 0:
            ind:Individ_T
            self.population=initial_population
            # Add individuals of appropriate types to the population
            for ind in initial_population:
                if ind.ind_type not in self.typecount_dict:
                    self.typecount_dict[ind.ind_type]   = 1
                else:
                    self.typecount_dict[ ind.ind_type ]+= 1

            self.poplen = reduce(lambda x,y: x+y, list(self.typecount_dict.values()))
            fitness_values        =  [*map(lambda individ: individ.fitness, self.population)]
            fitness_total         =  np.sum(fitness_values)
            self.average_fitness  =  fitness_total / self.poplen
            self.brate  =  ( self.average_fitness )/( self.average_fitness + self.poplen * BRATE_DENOM )
            self.drate  =  1 - self.brate

    def birth_death_event(self,current_iteration:int,)->None:

        if current_iteration == 0 or not current_iteration%resetcounterat:
            for individual in self.population:
                individual.calculate_fitness(self.fitmap.getMap())

        fitness_values        =  [*map(lambda individ: individ.fitness, self.population)]
        fitness_total         =  np.sum(fitness_values)
        self.average_fitness  =  fitness_total / self.poplen
        self.brate            =  ( self.average_fitness )/( self.average_fitness + self.poplen * BRATE_DENOM)
        self.drate            =  1 - self.brate

        normalized_fitness  =  [*map(lambda x : x / fitness_total, fitness_values) ]
        pick                =  np.random.choice([1, -1], p=[self.brate, self.drate])

        if pick > 0:
            chosen:Individ_T =  np.random.choice(self.population, p=normalized_fitness)
            chosen.give_birth(self, current_iteration)
        else:
            chosen = np.random.choice(self.population,replace=False)
            chosen.death(self)

    def add_individual(self,_type:int,ind:Individ_T,):
            self.population.append(ind)
            self.typecount_dict[_type] +=1
            self.poplen                +=1

    def remove_dead(self, _type:int, ind:Individ_T):
            self.population.remove(ind)
            self.typecount_dict[_type] -=1
            self.poplen                -= 1
    
    def updateFitmap(self,**kwargs):
        self.fitmap.mean       =  kwargs['mean'] if 'mean' in kwargs else self.fitmap.mean
        self.fitmap.amplitude  =  kwargs['amplitude'] if 'amplitude' in kwargs else self.fitmap.amplitude
        self.fitmap.std        =  kwargs['std'] if 'std' in kwargs else self.fitmap.std
        # print("Fimap is now std{} | mean{} | ampli {}".format(self.fitmap.std,self.fitmap.mean,self.fitmap.amplitude))

class Individ_T:
    """
    Abstract class for the individual types. 
    Although GPmaps, mutations, number of traits and genes can be arbitrary, 
    the type-tag is there to keep the track of them in the population.
    """

    def __init__(self, alleles:np.ndarray, gp_map:GPMap, ind_type:int):
        self.ind_type   =  ind_type
        self.alleles    =  alleles
        self.gp_map     =  gp_map
        self.phenotype  =  []
        self.fitness    =  0
        self.n_traits   =  np.shape(gp_map.coeffs_mat)[0]  # the number of traits is defined by the first dimension of the contribution matrix
        self.init_phenotype()

    def init_phenotype(self)-> List[float]:
        self.phenotype = self.gp_map.map_phenotype()
        return self.gp_map.map_phenotype()

    def calculate_fitness(self, fitmapfunc) -> float:
        """A single individual's fitness."""
        self.fitness= fitmapfunc(self.phenotype )
        return self.fitness

    def give_birth(self, population:Population, fitness_peak:int)->Individ_T:

        # TODO: GPMap lambda. Only the alleles are varied currently. 
        # TODO: Mutating GPMap would amount to tweaking degrees_mat and coeffs_map on self.GPMap
        """
        @ mutate_allele is applied to each allele with p(MUTATION_RATE)
        """
        def mutation_allele():
            return np.random.choice(MUTATION_VARIANTS_ALLELE)
        
        def mutation_duplicate(n_traits, alleles, gene_pos, coeff_matrix, deg_matrix):
            # ? where gene position is the index of the"column" of that genes contributions
            # --------------- Extending contribs

            duplicate_coeffs  =  coeff_matrix[:,gene_pos]
            duplicate_coeffs  =  np.reshape(duplicate_coeffs, (n_traits,1))
            coeff_extended    =  np.append(coeff_matrix, duplicate_coeffs , axis=1)

            duplicate_degs    =  deg_matrix[:,gene_pos]
            duplicate_degs    =  np.reshape(duplicate_degs, (n_traits,1))
            degs_extended     =  np.append(deg_matrix, duplicate_degs , axis=1)

            # Extending alleles
            newalleles        =  np.append(alleles, alleles[gene_pos])

            return [
            newalleles,
            degs_extended,
            coeff_extended
            ]
            
        def mutation_change_contrib(gene_pos, coeff_matrix,deg_matrix):
            """change of contribution pattern. genes can either lose or acquire a new function."""
            deg_contrib   =  deg_matrix[:, gene_pos]
            coef_contrib  =  coeff_matrix[:, gene_pos]

            def mutate_entry_contribution(entry:float)->float:
                return np.random.choice(np.arange(-1,2,1))

            coeff_matrix[:,gene_pos]   =  np.vectorize(mutate_entry_contribution)(coef_contrib)
            deg_matrix   [:,gene_pos]  =  deg_contrib

            return [deg_matrix, coeff_matrix]

        #! Mutation rats are defined at the top


        # template alleles from parent
        alleles_copy  =  np.copy(self.alleles)
        coeffs_copy   =  np.copy(self.gp_map.coeffs_mat)
        degs_copy     =  np.copy(self.gp_map.degrees_mat)

        #template alleles suffer a mutation
        did_mutate = False

        for index, gene in enumerate( self.alleles.tolist() ):
            if np.random.uniform() <= MUTATION_RATE_ALLELE:
                did_mutate = True
                alleles_copy[index] = mutation_allele()

            if np.random.uniform() <= MUTATION_RATE_DUPLICATION:
                did_mutate    =  True
                _             =  mutation_duplicate(self.n_traits, alleles_copy, index, coeffs_copy, degs_copy)
                alleles_copy  =  _[0]
                degs_copy     =  _[1]
                coeffs_copy   =  _[2]

            if np.random.uniform() <= MUTATION_RATE_CONTRIB_CHANGE:
                did_mutate   =  True
                _            =  mutation_change_contrib(index,coeffs_copy, degs_copy)
                degs_copy    =  _[0]
                coeffs_copy  =  _[1]


        if did_mutate:
            # gpmap is reinitialized based on the mutated genes
            newGPMap = GPMap(alleles_copy, self.gp_map.trait_n, self.gp_map.deg)
            newGPMap.deg_init(degs_copy)
            newGPMap.coef_init(coeffs_copy)
            nascent = Individ_T(alleles_copy, newGPMap, self.ind_type)

        else:
            # if haven't mutated -- gpmap is remains the same as parent
            nascent = Individ_T(self.alleles, self.gp_map, self.ind_type)

        # a new individual is born, inherits the type.
        population.add_individual(self.ind_type, nascent)
        return nascent

    def death(self, population:Population):
        population.remove_dead(self.ind_type,self)


def createIdividual(dicttype:str, ind_type)->Individ_T:
    inits  =  INDIVIDUAL_INITS[dicttype]
    gpmap  =  GPMap(inits['alleles'],inits['trait_n'], DEGREE);
    gpmap.coef_init(custom_coeffs=inits['coefficients'])
    return Individ_T(inits['alleles'], gpmap,ind_type)


POPULATION = [ ]
for _ in range(800):
    POPULATION.append(createIdividual("1.4",1))

_Fitmap            =  Fitmap(amplitude=1,std=1,mean=0)
population_proper  =  Population(_Fitmap,initial_population=POPULATION)
t1                 =  []
t6                 =  []
fitness            =  []
brate              =  []
fitpeak            =  []

#-⋯⋯⋅⋱⋰⋆⋅⋅⋄⋅⋅∶⋅⋅⋄▫▪▭┈┅✕⋅⋅⋄⋅⋅✕∶⋅⋅⋄⋱⋰⋯⋯⋯⋯⋅⋱⋱⋰⋆⋅⋅⋄⋅⋅∶⋅⋅⋄▫▪▭┈┅✕⋅⋅⋄⋅⋅✕∶⋅⋅⋄⋱⋰⋯⋯⋯⋅⋱⋰⋆⋅⋅⋄⋅⋅∶⋅⋅⋄▫▪▭┈┅✕⋅⋅⋄⋅⋅✕∶⋅⋅⋄⋱⋰⋯⋯⋯⋰⋆⋅⋅⋄⋅⋅∶⋅⋅⋄▫▪▭┈┅✕⋅⋅⋄⋅⋅✕∶⋅⋅⋄⋱⋰⋯⋯⋯⋅⋱⋰⋆⋅⋅⋄⋅⋅∶⋅⋅⋄▫▪▭┈┅✕⋅⋅⋄⋅⋅✕∶⋅⋅⋄⋱⋰⋯⋯⋯

index           =  -1
peak_val        =  0
peak_vals       =  [1,0,-1, 0]

t1.append(population_proper.typecount_dict[1])
t6.append(population_proper.typecount_dict[6])
fitness.append(population_proper.average_fitness)
brate.append(population_proper.brate)
fitpeak.append(population_proper.fitmap.mean)

for it in range(itern):

    t1.append(population_proper.typecount_dict[1])
    t6.append(population_proper.typecount_dict[6])
    fitness.append(population_proper.average_fitness)
    brate.append(population_proper.brate)
    fitpeak.append(population_proper.fitmap.mean)

    if ( not (it + 1) & (resetcounterat - 1) ) and SHIFTING_FITNESS_PEAK:
        index = index + 1
        population_proper.updateFitmap(mean=( population_proper.fitmap.mean+0.25 ))
    population_proper.birth_death_event(it)



for folder in ['fit', 'brate','t2','t1','t6','fitpeak' ]:
    os.makedirs(os.path.join(outdir,folder), exist_ok=True)

with open(os.path.join(outdir,'fitpeak','fitpeak_i{}.csv'.format(instance)), 'w',newline='') as filein:
    writer = csv.writer(filein)
    writer.writerows([fitpeak])
with open(os.path.join(outdir,'t1','t1_i{}.csv'.format(instance)), 'w',newline='') as filein:
    writer = csv.writer(filein)
    writer.writerows([t1])
with open(os.path.join(outdir,'t6','t6_i{}.csv'.format(instance)), 'w',newline='') as filein:
    writer = csv.writer(filein)
    writer.writerows([t6])

with open(os.path.join(outdir,'fit','fit_i{}.csv'.format(instance)), 'w',newline='') as filein:
    writer = csv.writer(filein)
    writer.writerows([fitness])
with open(os.path.join(outdir,'brate','brate_i{}.csv'.format(instance)), 'w',newline='') as filein:
    writer = csv.writer(filein)
    writer.writerows([brate])



if toplot:
    time = np.arange(len(fitness))
    figur, axarr = plt.subplots(4)
    axarr[0].plot(time, t1, label="Type 1", color="blue")
    axarr[0].plot(time, t6, label="Type 6", color="pink")
    axarr[0].set_ylabel('Individual Count')
    axarr[0].legend()

    axarr[1].plot(time, fitness, label="Fitness")
    axarr[1].set_ylabel('Populationwide Fitness')

    axarr[2].plot(time, brate, label="Birthrate")
    axarr[2].set_ylabel('Birthrate')

    axarr[3].plot(time, fitpeak, label="Fitness Peak")
    axarr[3].set_ylabel('Peak')

    figure = plt.gcf()
    figure.set_size_inches(12, 6)
    figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
    plt.show()