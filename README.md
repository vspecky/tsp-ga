# TSP with Genetic Algorithm
An assignment for my AI course with a solution to the TSP using a Genetic Algorithm

## How to Use:
```
python main.py help
```

## Reproduction Methodology
There are two types of Reproduction Modes implemented :-
- **Crossover**: In this method, two parents are chosen at random with a higher probability being given to more fit genomes. A random continuous subset of genes from the fitter parent is moved into the child, and the rest of the genes are filled in from the less fit parent. The child is then mutated according to a mutation rate.
- **Mutation**:  In this method, a single parent is randomly chosen with a higher probability being given to more fit genomes. The genes of the parent are copied over into the child and the child is mutated. Mutation involved swapping any two randomly selected genes in the gene pool.
