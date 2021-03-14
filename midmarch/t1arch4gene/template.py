from __future__ import annotations
from functools import reduce
import csv
import time
from operator import xor
import sys, os
import numpy as np
from typing import  Callable, List, TypedDict, Dict
import math


MUTATION_RATE_ALLELE          =  0.0001
"""The range of values that genes can take on. Declare global to not recreate on every mutation"""
MUTATION_VARIANTS_ALLELE      =  np.arange(-0.5,0.5005,0.005)
MUTATION_RATE_DUPLICATION     =  0.00
MUTATION_RATE_CONTRIB_CHANGE  =  0.00




def FITMAP(x):
    curve_height     =  10
    center_position  =  0
    std_dev          =  15
    return curve_height*math.exp(-(sum(((x - center_position)**2)/(2*std_dev**2))))

DEGREE = 1



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

    def peek(self):
        print("\n-----------------------")
        # print("Inited degrees to \t\n",self.degrees_mat)
        print("Inited coeffs to \t\n", self.coeffs_mat)

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
    def __init__(self, initial_population=[]):

        self.population:List[Individ_T]     = []
        self.typecount_dict                 = {}
        self.poplen: int                    = 0
        self.average_fitness:float          = 0
        self.brate: float                   = 0
        self.drate: float                   = 0

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

            self.brate               =  ( self.average_fitness )/( self.average_fitness + self.poplen * 0.01 )
            self.drate               =  ( self.poplen * 0.01 ) / (self.average_fitness + self.poplen*0.01)


    def birth_death_event(self)->None:

        #! Have to add to fitness total for each individual.
        #? major gains here with reducing the updates
        #? alternative to normalized picking too

        fitness_values        =  [*map(lambda individ: individ.fitness, self.population)]
        fitness_total         =  np.sum(fitness_values)
        self.average_fitness  =  fitness_total / self.poplen

        self.brate               =  ( self.average_fitness )/( self.average_fitness + self.poplen * 0.01 )
        self.drate               =  ( self.poplen * 0.01 ) / (self.average_fitness + self.poplen*0.01)

        normalized_fitness  =  [*map(lambda x : x / fitness_total, fitness_values) ]
        event               =  np.random.choice([1, -1], p=[self.brate, self.drate])

        if event > 0:
            chosen:Individ_T =  np.random.choice(self.population, p=normalized_fitness)
            chosen.give_birth(self)
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

class Individ_T:
    """
    Abstract class for the individual types. 
    Although GPmaps, mutations, number of traits and genes can be arbitrary, 
    the type-tag is there to keep the track of them in the population.
    """

    def __init__(self, alleles:np.ndarray, gp_map:GPMap, ind_type:int):
        self.ind_type  = ind_type
        self.alleles   = alleles
        self.gp_map    = gp_map
        self.phenotype = []
        self.fitness   = 0

        self.init_phenotype()
        self.calculate_fitness(FITMAP) #! FITMAP is globally defined

    # where fitmap is the function particular to each individual's type
    def init_phenotype(self)-> List[float]:


        self.phenotype = self.gp_map.map_phenotype()
        return self.gp_map.map_phenotype()

    def calculate_fitness(self,fitfunc) -> float:
        """A single individual's fitness."""

        self.fitness= fitfunc(self.phenotype)
        return self.fitness


    def give_birth(self, population:Population)->Individ_T:

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
            coeff_extended]
            
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
        alleles_copy  =  self.alleles.copy()
        coeffs_copy   =  self.gp_map.coeffs_mat.copy()
        degs_copy     =  self.gp_map.degrees_mat.copy()

        #template alleles suffer a mutation
        did_mutate = False

        for index, gene in enumerate( self.alleles.tolist() ):
            if np.random.uniform() <= MUTATION_RATE_ALLELE:
                did_mutate = True
                alleles_copy[index] = mutation_allele()
            if np.random.uniform() <= MUTATION_RATE_DUPLICATION:
                did_mutate    =  True
                _             =  mutation_duplicate(3, alleles_copy, index, coeffs_copy, degs_copy)
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



class IndividualType(TypedDict): 
      trait_n                  : int
      alleles                  : np.ndarray
      coefficients             : np.ndarray

INDIVIDUAL_INITS:Dict = {
   "1.1":{
        'trait_n' :3,
        'alleles'       :  np.array([1, 1, 0]),
        'coefficients'  :  np.array([
                        [1,0,0],
                        [0,1,0],
                        [0,0,1],
                    ])
   },
   "1.2":{
        'trait_n' :2,
        'alleles'       :  np.array([0.5, 0.5]),
        'coefficients'  :  np.array([
                        [1,0],
                        [0,1],
                    ])
   },
   "1.4":{
        'trait_n' :4,
        'alleles'       :  np.array([0.5, 0.5,0, 0]),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ])
   },
   "2":{
       'trait_n':3,
        'alleles'       :  np.array([1, 0, 0, 0 ,1 ,0 , 0,0,0]),
        'coefficients'  :  np.array([
                        [1,1,1,0,0,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0],
                        [0,0,0,0,0,0,1,1,1],
                    ])

   },
   "3":{
       "trait_n":3,
        'alleles'       :  np.array([1, 0, 0, 1]),
        'coefficients'  :  np.array([
                        [1,0,0,1],
                        [0,1,0,1],
                        [0,0,1,1],
                    ])

   },
   "4":{
   },
   "5":{"trait_n":4,
        'alleles'       :  np.array([1, 0,1]),
        'coefficients'  :  np.array([
                        [1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [1,1,1],
                        ])

   },
   "6":{
       "trait_n"          :  4,
        'alleles'       :  np.array([1,1]),
        'coefficients'  :  np.array([
                            [1,0],
                            [1,0],
                            [0,1],
                            [0,1],
                            ])
   }
}


def createIdividual(dicttype:str, ind_type)->Individ_T:
    inits  =  INDIVIDUAL_INITS[dicttype]
    gpmap  =  GPMap(inits['alleles'],inits['trait_n'], DEGREE);
    gpmap.coef_init(custom_coeffs=inits['coefficients'])
    return Individ_T(inits['alleles'], gpmap,ind_type)

POPULATION = [ ]

for _ in range (400):
    POPULATION.append(createIdividual("1.4",1))
    POPULATION.append(createIdividual("6",6))

population_proper  =  Population(initial_population=POPULATION)
siminst            =  sys.argv[1]

t1      =  []
t6      =  []

for it in range(int(3e6)):
    t1.append(population_proper.typecount_dict[1])
    t6.append(population_proper.typecount_dict[6])
    population_proper.birth_death_event()


os.makedirs('yield/t1')
os.makedirs('yield/t6')

with open(f'yield/t1/t1_siminst_{siminst}.csv', 'w',newline='') as filein:
    writer = csv.writer(filein)
    writer.writerows([t1])

with open(f'yield/t6/t6_siminst_{siminst}.csv', 'w',newline='') as filein:
    writer = csv.writer(filein)
    writer.writerows([t6])




