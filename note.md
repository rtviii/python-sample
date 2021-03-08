#### Let's optimize the fuck out of  this

- Integrate the BD-process into the population class. (Got only marginal speedups)

- moved poplen and fitness calculations to Population class. cut down from 535sec/10k to 13sec/10k. Embarassing.

- sample average fitness only every N iterations: 

    births and deaths of single individuals affect the measure only slightly while calculating it is an O(2n) operation -- plucking fitnesses and summing up

- rewrite the whole thing in cpp?

# Time profiling


