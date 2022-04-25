from itertools import product
import numpy as np
import matplotlib.pyplot as plt


N = 11
probs = [0.6]*10+[0.8]


def non_uniform_maj_vote(ps, n, weights=None):
    assert len(ps) == n
    if weights is None:
        weights = [1]*n
    weights = np.array(weights)
    weights = weights / weights.sum(0)
    prod = filter(lambda x: sum(weights*x) > 0.5, product([0,1],repeat=n))
    return sum([np.prod(np.array([ps[i] if p[i] else 1-ps[i] for i in range(n)])) for p in prod])


# print(non_uniform_maj_vote([0.6]*10, 10))
# print(non_uniform_maj_vote(probs, N))
# print(non_uniform_maj_vote(probs,N,[1]*10+[3]))

DEPTH = 4
STEP = 1/10**DEPTH

from multiprocessing import Pool

with Pool(7) as p:
    plt.plot(np.arange(0,10,STEP), p.starmap(non_uniform_maj_vote, [ (lambda n: (probs[:n],n,[1]*10+([i] if i > 0 else [])))(N if i > 0 else N-1) for i in np.arange(0,10,STEP) ]))
    plt.show()

# plt.plot(np.arange(0,10,STEP), [ (lambda n: non_uniform_maj_vote(probs[:n],n,[1]*10+([i] if i > 0 else [])))(N if i > 0 else N-1) for i in np.arange(0,10,STEP) ])
# plt.show()
