import sys
import os

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def read_data(filepath):
    '''Read data points from file specified by filepath
    Args:
        filepath (str): the path to the file to be read

    Returns:
        numpy.ndarray: a numpy ndarray with shape (n, d) where n is the number of data points and d is the dimension of the data points

    '''

    X = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            X.append([float(e) for e in line.strip().split(',')])
    return np.array(X)

def dist_func(point_1, point_2,metric="euclidean"):
    """
        This implements the distance function.
    """
    return metrics.pairwise_distances(point_1, point_2, metric=metric)[0][0]

def range_query(X,P,eps):
    """
        This function takes the DB, a point P, and returns a set
        of tuples of the form (Q, idx), where idx is the index into
        the original databases.
    """
    neighbours = set([])
    for idx, point in enumerate(X):
        if dist_func(np.array([point]), np.array([P])) <= eps:
            neighbours.add((tuple(point.tolist()), idx))
    return neighbours

def difference(points, N):
    return points.difference(set([N]))

# To be implemented
def dbscan(X, eps, minpts):
    '''dbscan function for clustering
    Args:
        X (numpy.ndarray): a numpy array of points with dimension (n, d) where n is the number of points and d is the dimension of the data points
        eps (float): eps specifies the maximum distance between two samples for them to be considered as in the same neighborhood
        minpts (int): minpts is the number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself.
    
    Returns:
        list: The output is a list of two lists, the first list contains the cluster label of each point, where -1 means that point is a noise point, the second list contains the indexes of the core points from the X array.
    
    Example:
        Input: X = np.array([[-10.1,-20.3], [2.0, 1.5], [4.3, 4.4], [4.3, 4.6], [4.3, 4.5], [2.0, 1.6], [2.0, 1.4]]), eps = 0.1, minpts = 3
        Output: [[-1, 1, 0, 0, 0, 1, 1], [1, 4]]
        The meaning of the output is as follows: the first list from the output tells us: X[0] is a noise point, X[1],X[5],X[6] belong to cluster 1 and X[2],X[3],X[4] belong to cluster 0; the second list tell us X[1] and X[4] are the only two core points

        '''
    print("Size of the array: ", X.shape)
    #Initializing label numpy array with zeros.
    # 0 -> undefined
    # 1 -> core
    # 2 -> Noise
    C = 0
    label = np.zeros(X.shape[0])
    for idx, point in enumerate(X):
        print(idx)
        if label[idx] != 0:
            continue
        
        neighbours = range_query(X, point, eps)
        if len(neighbours) < minpts:
            label[idx] = 2
            continue

        C = C + 1
        label[idx] = C

        seed_set = difference(neighbours, (tuple(point.tolist()), idx))
        while seed_set:
            Q = seed_set.pop()
            print(Q)
            if label[Q[1]] == 2:
                label[Q[1]] = C
            if label[Q[1]] != 0:
                continue

            label[Q[1]] = C
            neighbours = range_query(X, np.array(Q[0]), eps)
            if len(neighbours) >= minpts:
                seed_set.update(neighbours)
    print(label)
    return [[], []]




def main():
    
    if len(sys.argv) != 4:
        print("Wrong command format, please follwoing the command format below:")
        print("python dbscan-template.py data_filepath eps minpts")
        exit(0)

    X = read_data(sys.argv[1])

    # Compute DBSCAN
    db = dbscan(X, float(sys.argv[2]), int(sys.argv[3]))

    # store output labels returned by your algorithm for automatic marking
    with open('.'+os.sep+'Output'+os.sep+'labels.txt', "w") as f:
        for e in db[0]:
            f.write(str(e))
            f.write('\n')

    # store output core sample indexes returned by your algorithm for automatic marking
    with open('.'+os.sep+'Output'+os.sep+'core_sample_indexes.txt', "w") as f:
        for e in db[1]:
            f.write(str(e))
            f.write('\n')

    _,dimension = X.shape

    # plot the graph is the data is dimensiont 2
    if dimension == 2:
        core_samples_mask = np.zeros_like(np.array(db[0]), dtype=bool)
        core_samples_mask[db[1]] = True
        labels = np.array(db[0])

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]


        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig('.'+os.sep+'Output'+os.sep+'cluster-result.png')

main()
