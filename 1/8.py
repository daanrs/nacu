from deap import base, creator, gp, algorithms, tools

import math
import operator

import numpy as np

# input
inp = [-1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. ,
       0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]

#output
outp = [
    0, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784,
    -0.2289, -0.1664, -0.0909, 0, 0.1111, 0.2496, 0.4251,
    0.6496, 0.9375, 1.3056, 1.7731, 2.3616, 3.0951, 4
]

points = list(zip(inp, outp))

# code is partially referenced from the DEAP GP documentation
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedLog(x):
    try:
        return math.log(x)
    except ValueError:
        return 0

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.exp, 1)

pset.renameArguments(ARG0="x")

creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    a = np.array([func(x) - y for x, y in points])
    return np.sum(np.abs(a)),

toolbox.register("evaluate", evalSymbReg, points=points)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut,
                 pset=pset)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

def main():
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(50)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0, 50, halloffame=hof,
                                   stats=mstats, verbose=True)
