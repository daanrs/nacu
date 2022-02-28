from random import uniform, choice
import numpy as np
from math import dist
import matplotlib.pyplot as plt


def genAD1():
    data_vectors = []
    for _ in range(400):
        z1 = uniform(-1, 1)
        z2 = uniform(-1, 1)
        c = int(z1 >= 0.7 or z1 <= 0.3 and z2 >= -0.2 - z1)
        data_vectors.append((np.array([z1, z2]), c))
    return data_vectors, 2, 400, 2  # ..., Nd, No, Nc

def getIrisData():
    f = open("../iris.data", "r")
    data_vectors = []
    for d in [x.split(',') for x in f.read().split()]:
        c = ["setosa", "versicolor", "virginica"].index(d[4].split('-')[-1])
        data_vectors.append((np.array(list(map(float, d[:4]))), c))
    return data_vectors, 4, len(data_vectors), 3



# Nd = 3  # Input dimension
# No = 20  # Number of data vectors to be clustered
# Nc = 5  # Number of cluster centroids

def select_random_cluster_centroids(dv, n):
    centroids = []
    cs = []
    i = 0
    while i < n:
        c = choice(dv)
        if c[1] in cs:
            continue
        else:
            centroids.append(c[0])
            cs.append(c[1])
            i += 1
    return np.array(centroids)


class Particle:
    def __init__(self, centroids, dims, w=0.7298, c1=1.49618, c2=1.49618):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dims = dims
        self.N_CLUSTERS = len(centroids)

        self.current = (np.copy(centroids), None)
        self.velocity = np.array([0] * dims)*self.N_CLUSTERS
        self.local_best = (np.copy(centroids), None)

        self.init_clusters()

    def init_clusters(self):
        self.clusters = [[] for _ in range(self.N_CLUSTERS)]

    def compute_fitness(self):
        sum1 = 0
        for i in range(self.N_CLUSTERS):
            if len(self.clusters[i]) != 0:
                sum2 = sum([dist(zp, self.current[0][i]) for zp in self.clusters[i]])
                sum1 += sum2 / len(self.clusters[i])
            else:
                sum1 += float("inf")
        fitness = sum1 / self.N_CLUSTERS
        self.current = self.current[0], fitness
        return fitness

    def update_vel(self, global_best):
        for i in range(self.dims):
            r1 = np.random.uniform(0, 1, (self.N_CLUSTERS, self.dims))
            r2 = np.random.uniform(0, 1, (self.N_CLUSTERS, self.dims))
            self.velocity = self.w * self.velocity \
                            + self.c1 * r1 * (self.local_best[0] - self.current[0]) \
                            + self.c2 * r2 * (global_best - self.current[0])

    def update_pos(self):
        self.current = self.current[0] + self.velocity, self.current[1]

    def update_local_best(self):
        if self.local_best[1] is None or self.current[1] < self.local_best[1]:
            self.local_best = self.local_best[0], self.current[1]

    def update(self, gb):
        self.update_vel(gb)
        self.update_pos()


class PSOSwarm:
    def __init__(self, data_vectors, nd, nc, num_particles=10):
        self.data_vectors = data_vectors
        self.particles = []
        for i in range(num_particles):
            centroids = select_random_cluster_centroids(data_vectors, nc)
            self.particles.append(Particle(centroids, nd))
        self.global_best = (np.array([0] * nd), None)
        self.best_clustering = None
        self.plot_data = []

    def run(self, t_max, prints=False, plot=True):
        for t in range(t_max):
            for p in self.particles:
                p.init_clusters()
                for zp in [x for x, _ in self.data_vectors]:
                    dists = [dist(zp, m) for m in p.current[0]]
                    p.clusters[dists.index(min(dists))].append(zp)  # Append zp to the cluster that has the closest centroid
                fitness = p.compute_fitness()
                p.update_local_best()
                if self.global_best[1] is None or fitness < self.global_best[1]:
                    self.global_best = p.current
                    self.best_clustering = p.clusters

            for p in self.particles:
                p.update(self.global_best[0])

            if prints:
                print("Fitness at time", t, ':', self.global_best[1])
            self.plot_data.append(self.global_best[1])
            if plot:
                plt.plot(self.plot_data, color='r')

        if prints:
            print("\nOptimal solution:", self.global_best[0])
            print("Optimal fitness:", self.global_best[1])
        # diff = max(self.plot_data) - min(self.plot_data)
        # plt.ylim(min(self.plot_data)-diff/10, max(self.plot_data)+diff/10)
        if plot:
            plt.show()
        return self.plot_data, self.best_clustering


class KMeans:
    def __init__(self, data_vectors, nd, nc):
        self.data_vectors = data_vectors
        self.dims = nd
        self.N_CLUSTERS = nc
        self.particles = []
        self.centroids = select_random_cluster_centroids(data_vectors, nc)
        self.plot_data = []
        self.clusters = [[] for _ in range(nc)]

    def recompute_centroids(self):
        for i in range(self.N_CLUSTERS):
            self.centroids[i] = np.mean(self.clusters[i], axis=0)

    def compute_fitness(self):
        sum1 = 0
        for i in range(self.N_CLUSTERS):
            if len(self.clusters[i]) != 0:
                sum2 = sum([dist(zp, self.centroids[i]) for zp in self.clusters[i]])
                sum1 += sum2 / len(self.clusters[i])
        return sum1 / self.N_CLUSTERS

    def run(self, t_max, plot=True):
        for t in range(t_max):
            self.clusters = [[] for _ in range(nc)]
            for zp in [x for x, _ in self.data_vectors]:
                dists = [dist(zp, m) for m in self.centroids]
                self.clusters[dists.index(min(dists))].append(zp)
            self.recompute_centroids()
            fitness = self.compute_fitness()
            self.plot_data.append(fitness)
            if plot:
                print("Fitness at time", t, ':', fitness)
                plt.plot(self.plot_data, color='r')
        if plot:
            print("\nOptimal solution:", self.centroids)
            print("Optimal fitness:", self.plot_data[-1])
            # diff = max(self.plot_data) - min(self.plot_data)
            # plt.ylim(min(self.plot_data) - diff / 10, max(self.plot_data) + diff / 10)
            plt.show()
        return self.plot_data, self.clusters

    def runNTimes(self, t_max, N, plot=True):
        plot_datas = []
        for i in range(N):
            k = KMeans(self.data_vectors, self.dims, self.N_CLUSTERS)
            plot_datas.append(k.run(t_max, plot=False)[0])
        mean = np.mean(plot_datas, axis=0)
        if plot:
            print("Optimal fitness:", mean[-1])
            plt.plot(mean, color='r')
            plt.show()
        return mean


# Only works for 2 dimensional vectors (like Artificial Dataset 1)
def clusterPlot(clusters, data_vectors):
    print(clusters)
    for cl, c in zip(clusters, ['blue', 'orange']):
        s1 = [x[0] for x in cl]
        s2 = [x[1] for x in cl]
        plt.plot(s1, s2, 'o', color=c)
    truePlot(data_vectors)


def truePlot(data_vectors):
    s1 = ([], [])
    s2 = ([], [])
    for dv, c in data_vectors:
        if c == 0:
            s1[0].append(dv[0])
            s1[1].append(dv[1])
        else:
            s2[0].append(dv[0])
            s2[1].append(dv[1])
    plt.plot(s1[0], s1[1], 'o', alpha=0.25, color='black')
    plt.plot(s2[0], s2[1], 'o', alpha=0.25, color='white')
    plt.show()


if __name__ == "__main__":
    # num_particles = 10
    T_MAX = 100
    TRIALS = 30

    data_vectors, nd, _, nc = genAD1()
    # data_vectors, nd, _, nc = getIrisData()

    s = PSOSwarm(data_vectors, nd, nc)
    clusters = s.run(T_MAX)[1]
    clusterPlot(clusters, data_vectors)

    k = KMeans(data_vectors, nd, nc)
    clusters = k.run(T_MAX)[1]
    clusterPlot(clusters, data_vectors)
    # k.runNTimes(T_MAX, 10)

    RUN = "PSO"
    # RUN = "KMeans"

    print("Running %s for %d trials with %d iterations" % (RUN, TRIALS, T_MAX))

    plot_datas = []
    for i in range(TRIALS):
        if RUN == "PSO":
            s = PSOSwarm(data_vectors, nd, nc)
            pd, _ = s.run(T_MAX, plot=False)
        else:
            k = KMeans(data_vectors, nd, nc)
            pd = k.runNTimes(T_MAX, 10, plot=False)
        plot_datas.append(pd)

    print("Optimal fitness:", np.mean(plot_datas, axis=0)[-1])


