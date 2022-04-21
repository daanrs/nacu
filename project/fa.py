import numpy as np

from scipy.spatial import distance_matrix


def firefly(xs, score, alpha=0.5, beta=1, gamma=1, max_iter=1000):
    for n in range(max_iter):
        scores = np.array([score(x) for x in xs])

        r = distance_matrix(xs, xs)
        diff = xs[:, np.newaxis, ...] - xs[np.newaxis, ...]

        # TODO: why axis=0?
        attraction_movement = np.sum(
            np.exp(-1 * gamma * (r ** 2)) * diff,
            axis=0
        )
        mask = np.less.outer(scores, scores)
        attraction_movement = np.where(mask, attraction_movement, 0)

        random_walk = np.random.random_sample(xs.shape) - 0.5

        xs = (
            xs
            + beta * attraction_movement
            + alpha * random_walk
        )
    scores = np.array([score(x) for x in xs])
    return xs[np.argmax(scores)]
