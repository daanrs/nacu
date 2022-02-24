# Imports
from math import dist
from numpy.random import permutation
from random import randrange as rr, sample
import matplotlib.pyplot as plt
import sys
import multiprocessing

class TSPSolver:
    def __init__(self, cities, pop_size=20, mutation_probability=0.05, use_2opt=False, cores=8, plot=False):
        self.cities = cities
        self.N = len(cities)
        self.pop_size = pop_size
        self.pm = mutation_probability
        self.use_2opt = use_2opt
        self.cores = cores
        self.plot = plot
        
        
    # Define the fitness function
    def fitness(self, route):
        travelled = 0
        for i in range(self.N-1):
            travelled += dist(self.cities[route[i]],self.cities[route[i+1]])
        return 1/travelled

    def avg_dist_travelled(self, pop):
        return sum([ 1/self.fitness(x) for x in pop ])/self.pop_size

    def min_dist_travelled(self, pop):
        return min([ 1/self.fitness(x) for x in pop ])
    
    
    # Define the crossover function  | input:  two parents and optionally two cut points  
    #                                | output: two offspring
    def crossover(self, parent1, parent2, cuts=()):
        if cuts == ():
            c1, c2 = sorted(sample(range(self.N),2))
        else:
            c1 = cuts[0]; c2 = cuts[1]
        if c1 >= c2 or c1 < 0 or c2 > self.N or len(parent1) != self.N or len(parent2) != self.N:
            sys.exit("bad crossover parameters")

        p1 = [ x for x in parent1 ]
        p2 = [ x for x in parent2 ]
        o = [p1,p2]
        rest = (p1[:c1] + p1[c2:], p2[:c1] + p2[c2:])

        for i in [0,1]:
            s = [ x for x in rest[1-i][c1:] + rest[1-i][:c1] if x in rest[i] ] + [ x for x in rest[i] if x not in rest[1-i] ]
            o[i][c2:] = s[:self.N-c2]
            o[i][:c1] = s[self.N-c2:]
        return o
    
    
    # Define mutation  --  swap two random indices
    def mutate(self, s):
        r1, r2 = sample(range(self.N),2)
        x = s[r1]
        s[r1] = s[r2]
        s[r2] = x
        return s
    
    
    # Initialise random population
    def init(self):
        init_pop = []
        count = 0
        while count < self.pop_size:
            c = list(permutation(self.N))
            if c not in init_pop:
                init_pop.append(c)
                count += 1
        return init_pop
    
    
    # Binary tournament selection
    def BTS(self, pop):
        l = len(pop)
        random_order = sample(range(l),l)
        sel_pop = [ pop[a] if self.fitness(pop[a]) > self.fitness(pop[b]) else pop[b] for (a,b) in zip(random_order[::2],random_order[1::2]) ]
        if l % 2 == 1:
            sel_pop += [pop[random_order[-1]]]
        return sel_pop
    
    
    # Define the 2opt algorithm for the MA
    def two_opt_swap(self, route, i, k):
        return route[:i] + route[i:k+1][::-1] + route[k+1:]

    def two_opt(self, r):
        best = self.fitness(r)
        new = best
        while best == new:  # Reduce complexity to improve hybridisation
            for i in range(0,len(r)):
                found = False
                for k in range(i+1,len(r)):
                    new_route = self.two_opt_swap(r,i,k)
                    new = self.fitness(new_route)
                    if new > best:
                        r = new_route
                        best = new
                        found = True
                        break
                if found:
                    break
        return r
    
    
    # For multicore
    def compute(self,ps):
        (p1,p2) = ps
        c1,c2 = [ self.mutate(x) if rr(int(1/self.pm)) == 0 else x for x in self.crossover(p1,p2) ]
        return [p1,p2] + list(map(lambda c: self.two_opt(c) if self.use_2opt else c, [c1,c2]))
        
    # Define the single core offspring generation function
    def expand_popMC(self, pop):
        l = len(pop)
        random_order = [ pop[i] for i in sample(range(l),l) ]
        pool = multiprocessing.Pool(self.cores)
        new_pop = [ x for i in pool.map(self.compute, list(zip(random_order[::2],random_order[1::2]))) for x in i ]
        pool.close()
        pool.join()
        if l % 2 == 1:
            new_pop += [random_order[-1]] + ([] if self.pop_size % 2 == 1 else [self.mutate(random_order[-1])])
        return new_pop
    
    # Define the single core offspring generation function
    def expand_popSC(self, pop):
        l = len(pop)
        random_order = [ pop[i] for i in sample(range(l),l) ]
        new_pop = []
        for (p1,p2) in zip(random_order[::2],random_order[1::2]):
            c1,c2 = [ self.mutate(x) if rr(int(1/self.pm)) == 0 else x for x in self.crossover(p1,p2) ]
            new_pop += [p1,p2] + list(map(lambda c: self.two_opt(c) if self.use_2opt else c, [c1,c2]))
        if l % 2 == 1:
            new_pop += [random_order[-1]] + ([] if self.pop_size % 2 == 1 else [self.mutate(random_order[-1])])
        return new_pop
   
    def expand_pop(self, pop):
        if self.use_2opt:
            return self.expand_popMC(pop)
        else:
            return self.expand_popSC(pop)
      
    
    # Run TSPSolver
    def run(self, steps):
        pop = self.init()
        avg_dists = []; min_dists = []
        for i in range(steps):
            pop = self.expand_pop(self.BTS(pop))
            avg_dists.append(self.avg_dist_travelled(pop))
            min_dists.append(self.min_dist_travelled(pop))
    
        print("Minimum distance after " + str(steps) + " steps with " + ("M" if self.use_2opt else "E") + "A:", round(min(min_dists)))

        # Plot result    
        if self.plot:
            plt.plot(avg_dists, label=("MA" if self.use_2opt else "EA") + "-avg")
            plt.plot(min_dists, label=("MA" if self.use_2opt else "EA") + "-min")
            plt.legend()
        
        return [avg_dists, min_dists]
    
    
# Load cities from file
f1 = open('file-tsp.txt', 'r')
cities1 = list(map(lambda x: list(map(float,x.split())),f1.read().split('\n')))

f2 = open('bays29.tsp', 'r')
cities2 = list(map(lambda x: list(map(float,x.split()[1:])),f2.read().split('\n')[38:-2]))


# Exercise 6b

STEPS = 1500

inps = [(cities1,'file-tsp.txt'),(cities2,'bays29.tsp')]

for cities,fname in inps:
    print("\nRUNNING ON TSP FROM %s\n" % fname)
    avg_ea = []; min_ea = []
    avg_ma = []; min_ma = []

    ea = TSPSolver(cities); ma = TSPSolver(cities, use_2opt=True)
    for _ in range(10):
        a,m = ea.run(STEPS); avg_ea.append(a); min_ea.append(m)
        a,m = ma.run(STEPS); avg_ma.append(a); min_ma.append(m)

    # Plot EA result        
    plt.plot([ float(sum(l))/len(l) for l in zip(*avg_ea) ], label="EA-avg")
    plt.plot([ float(sum(l))/len(l) for l in zip(*min_ea) ], label="EA-min")
    plt.xlabel("Number of iterations"); plt.ylabel("Distance travelled")
    plt.legend(); plt.show()

    # Plot MA result
    plt.plot([ float(sum(l))/len(l) for l in zip(*avg_ma) ], label="MA-avg")
    plt.plot([ float(sum(l))/len(l) for l in zip(*min_ma) ], label="MA-min")
    plt.xlabel("Number of iterations"); plt.ylabel("Distance travelled")
    plt.legend(); plt.show()
