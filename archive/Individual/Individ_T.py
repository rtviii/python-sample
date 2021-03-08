from __future__ import annotations
from functools import reduce
import sys
import numpy as np
from typing import  Callable, List
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import in1d

def FITMAP(x):
    """Dummy for fitness function.
    Is defined globally. Should probably have a switch case for each of the individual types.
    """
    return np.sum(x)

N_TRAITS       =  3
DEGREE         =  1
MUTATION_RATE  =  0.05
POPULATION     =  []

class GPMap():    #! DEGREE is globally defined
    #! TRAIT_N is globally defined, traits are never permuted
    """
    - is tied to the individual type
    on birth, either:
        - altered on mutation
        - or inherited"""

    def __init__(self,alleles:List[float], trait_n:int, deg:int) -> None:
        #TODO: A way to inject variance, vary the maps
        #? g:= length of genome
        #? t:= number of traits
        # Both coefficient and degree matricies are t x g
        self.trait_n        =  trait_n
        self.deg            =  deg
        self.alleles        =  alleles
        self.genome_length  =  len(alleles)

        # Coefficients and degree matrices that constitute the gp_map
        self.coeffs_mat   =  None
        self.degrees_mat  =  None
        
        self.deg_init()
        self.coef_init()


    def coef_init(self, custom_coeffs=None):
        if custom_coeffs is not None: 
            self.coeffs_mat = custom_coeffs
            return
        # Coefficient initialization
        coeffs = np.full((self.trait_n, self.genome_length), fill_value=0)
        # Contributions are linear. There is no mutations
        for x in range(self.trait_n):
            coeffs[x][x] = 1
        self.coeffs_mat = coeffs
        return 

    def deg_init(self, custom_degrees=None):
        #? Degree initialization
        if custom_degrees is not None: 
            self.degrees_mat = custom_degrees
            return
        self.degrees_mat = np.full((self.trait_n,self.genome_length), fill_value=self.deg)
        return self.degrees_mat

    def peek(self):
        print("\tInited degrees to \n",self.degrees_mat)
        print("\tInited coeffs to \n", self.coeffs_mat)

    def map_phenotype(self)->List[float]:
        return  np.sum(self.coeffs_mat * ( self.alleles ** self.degrees_mat), axis=1)

# Abstract class for the 4 individual types
class Individ_T:
    #  where alleles are a vector of varying length(based on the type1,2,3,4), a genome
    def __init__(self, alleles:List[float], gp_map:GPMap, type:int):
        self.type       =  type
        self.alleles    =  alleles
        self.gp_map     =  gp_map
        self.phenotype  =  []
        self.fitness    =  0

        self.init_phenotype()
        self.calculate_fitness(FITMAP)

    # where fitmap is the function particular to each individual's type
    def init_phenotype(self)-> List[float]:
        """
        Instantiates alleles(however many of them might be) into traits
        @ gp_map  -- function that acts on allels. Must conform with the given individual Type(allele length).
        """
        self.phenotype = self.gp_map.map_phenotype()
        return self.gp_map.map_phenotype()

    def calculate_fitness(self,fitfunc) -> float:
        """An individual's fitness."""
        self.fitness= fitfunc(self.phenotype)
        return self.fitness

    def give_birth(self, population:List,
                    mutate_allele:Callable[[ float ], float] = None):
        """
        @ mutate_allele is applied to each allele with p(mutation_rate) -- globally defined
        """
        # TODO: GPMap lambda. not touching gpmap now
        # genomelength =len( self.alleles )

        def mutate_allele():
            new_variant = np.random.choice(np.arange(0,1.25,0.25))
            return new_variant
        
        newallels = self.alleles.copy()
        for index, gene in enumerate( self.alleles ):
            if np.random.uniform() <= MUTATION_RATE:
                newallels[index] = mutate_allele()

        nascent = Individ_T(newallels, self.gp_map, self.type)
        population.append(nascent)

    def death(self, population:List[Individ_T]):
        population.remove(self)

def birth_death_event(population:List[Individ_T]):
    fitness_values      =  [*map(lambda individ: individ.fitness, population)]
    normalized_fitness  =  [*map(lambda x : x /np.sum(fitness_values), fitness_values) ]
    brate               =  sum(fitness_values)/(sum(fitness_values) + len(population))
    drate               =  len(population)/(sum(fitness_values) + len(population))
    event               =  np.random.choice([brate, drate], p=[brate, drate])

    if event ==  brate:
        chosen:Individ_T      =  np.random.choice(population, p=normalized_fitness)
        chosen.give_birth(population)
    else:
        chosen = np.random.choice(population,replace=False)
        chosen.death(population)

    return population



# !assuming
# number of traits = const
# fitmap is globally defined
# gp_map degrees are globally defined (for now)


#  ------------------------------------------------------------
T1_alleles   =  np.array([0.2,0.4,0.6])
T1_coeff = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]])

T2_alleles   =  np.array([0.1,0.1,0.1,   0.3,0.3,0.3,  0.5,0.5,0.5])
T2_coeff = np.array([
    [1,1,1,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,1,1,1]])
#  ------------------------------------------------------------

T1_gpmap  =  GPMap(T1_alleles,N_TRAITS, DEGREE)
T1_gpmap.coef_init(custom_coeffs=T1_coeff)
i_T1      =  Individ_T(T1_alleles,T1_gpmap, 1)

T2_gpmap  =  GPMap(T2_alleles,N_TRAITS, DEGREE)
T2_gpmap.coef_init(custom_coeffs=T2_coeff)
i_T2      =  Individ_T(T2_alleles,T2_gpmap, 3)


for _ in range (250):
    POPULATION.append(i_T1)
    POPULATION.append(i_T2)

def gett1(pop:List[Individ_T]):
    sgm = 0
    for x in pop:
        if x.type==1:
            sgm+=1  
    return sgm

def gett3(pop:List[Individ_T]):
    sgm = 0
    for x in pop:
        if x.type==3:
            sgm+=1  
    return sgm

ITERN = int( sys.argv[1] )


for it in range(ITERN):
    birth_death_event(POPULATION)
    print(f"\t\t {gett1(POPULATION)}\t|\t{gett3(POPULATION)}")

# plt.plot(indt1,np.arange(500))
# plt.plot(indt3, np.arange(500))
# plt.show()