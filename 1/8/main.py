# code is partially referenced from the DEAP GP documentation
from deap import base, creator, gp, algorithms, tools

import math
import operator
import random

import numpy as np


# input
inp = (-1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. ,
       0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. )

#output
outp = (
    0, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784,
    -0.2289, -0.1664, -0.0909, 0, 0.1111, 0.2496, 0.4251,
    0.6496, 0.9375, 1.3056, 1.7731, 2.3616, 3.0951, 4
)

points = tuple(zip(inp, outp))

creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# use n=10
def main(n):

    random.seed(n)

    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.truediv, 2)
    pset.addPrimitive(math.log, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.exp, 1)

    pset.renameArguments(ARG0='x')

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalSymbReg(individual, points):
        """
        Evaluate the tree on our points. Return the sum of absolute errors
        (which we want to minimize).

        We have to adress a bunch of errors which can arise from domains of
        div and log, as well as infinite values which arise from overflows
        sometimes. We choose to return fitness=inf (so, infinitely bad) if
        any errors occur.
        """
        func = toolbox.compile(expr=individual)
        try:
            return np.sum(np.abs(np.array([func(x) - y for x, y in points]))),
        except ZeroDivisionError:
            return math.inf,
        except ValueError:
            return math.inf,
        except OverflowError:
            return math.inf,

    toolbox.register("evaluate", evalSymbReg, points=points)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut,
                     pset=pset)

    stats_fit = tools.Statistics(lambda ind: (ind.fitness.values[0],
                                              len(ind)))
    stats_fit.register('min', lambda x: x[np.argmin(np.array(x)[:, 0])])

    stats_size = tools.Statistics(len)
    stats_size.register('max', np.max)
    stats_size.register('min', np.min)
    stats_size.register('mean', np.mean)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    pop = toolbox.population(n=1000)
    # hof = tools.HallOfFame(1)
    return algorithms.eaSimple(pop, toolbox, 0.7, 0, 50, stats=mstats)
