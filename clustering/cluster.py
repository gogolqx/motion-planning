import pickle
import numpy as np
import pandas as pd
import os.path
import sys
import argparse
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score


def hierarchical_clustering(distance_matrix, n_clusters=None, proximity_measure='complete', distance_threshold=None):
    if distance_threshold:
        cluster = AgglomerativeClustering(
            affinity='precomputed',
            linkage=proximity_measure,
            distance_threshold=distance_threshold)
    elif n_clusters:
        cluster = AgglomerativeClustering(
            affinity='precomputed',
            linkage=proximity_measure,
            n_clusters=n_clusters)
    else:
        print("Please provide the number of clusters or a distance threshold")
        exit(1)
    return cluster.fit_predict(distance_matrix)


def dbscan_clustering(distance_matrix, eps=0.2, n_neighbours=5):
    cluster = DBSCAN(eps=eps, min_samples=n_neighbours, metric='precomputed')
    return cluster.fit_predict(distance_matrix)


def spectral_clustering(distance_matrix, n_clusters):
    cluster = SpectralClustering(
        n_clusters=n_clusters,
        eigen_solver='amg',
        affinity='precomputed',
        n_jobs=-1
    )
    similarity_matrix = 1 - distance_matrix
    return cluster.fit_predict(similarity_matrix)


parser = argparse.ArgumentParser(description='Cluster based on distance matrix')


class ClusteringCLI():

    def __init__(self):
        pass

    def run(self):
        parser = argparse.ArgumentParser(
            description='Cluster based on distance matrix',
            usage='''cluster <command> [<args>]

The commands are:
   dbscan           Run DBSCAN Clustering
   hierarchical     Run Hierarchical Clustering
   spectral         Run Spectral Clustering
''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        return getattr(self, args.command)()

    def dbscan(self):
        parser = argparse.ArgumentParser(
            description='Arguments for DBSCAN')
        parser.add_argument('dataset', type=str,
                            help='A csv file path')
        parser.add_argument('matrix', type=str,
                            help='A pkl file path to the distance matrix')
        parser.add_argument('-e', '--epsilon', default=0.2, type=float,
                            dest='epsilon',
                            help='The maximum distance so it is still considered in the same neighbourhood')
        parser.add_argument('-n', '--n_neighbours', default=5, type=int,
                            dest='n_neighbours',
                            help='Number of neighbours within a distance of epsilon so the point is considered an inner point')

        args = parser.parse_args(sys.argv[2:])
        matrix = np.array(pickle.load(open(args.matrix, 'rb')))
        return dbscan_clustering(matrix, args.epsilon, args.n_neighbours), matrix, args, \
               {'method':'DBSCAN','epsilon':args.epsilon, 'neighbours':args.n_neighbours}

    def hierarchical(self):
        parser = argparse.ArgumentParser(
            description='Arguments for Agglomerative Clustering')
        parser.add_argument('dataset', type=str,
                            help='A csv file path')
        parser.add_argument('matrix', type=str,
                            help='A pkl file path to the distance matrix')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('-n', '--n_clusters', default=6, type=int,
                            dest='n_clusters', help='Number of clusters')
        group.add_argument('-d', '--max_distance', type=float,
                           dest='max_distance', help='Distance threshold between clusters')
        parser.add_argument('-m', '--metric', default='complete', type=str,
                            dest='metric',
                            help='Distance metric between two clusters. Options are "ward", "single", "complete"')

        args = parser.parse_args(sys.argv[2:])
        matrix = np.array(pickle.load(open(args.matrix, 'rb')))
        return hierarchical_clustering(matrix, args.n_clusters, args.metric, args.max_distance), matrix, args, \
               {'method':'hierarchical', 'metric':args.metric}

    def spectral(self):
        parser = argparse.ArgumentParser(
            description='Arguments for Spectral Clustering')
        parser.add_argument('dataset', type=str,
                            help='A csv file path')
        parser.add_argument('matrix', type=str,
                            help='A pkl file path to the distance matrix')
        parser.add_argument('-n', '--n_clusters', default=6, type=int,
                           dest='n_clusters', help='Number of clusters')

        args = parser.parse_args(sys.argv[2:])
        matrix = np.array(pickle.load(open(args.matrix, 'rb')))
        return spectral_clustering(matrix, args.n_clusters), matrix, args, {'method': 'spectral'}


cli = ClusteringCLI()
clusters, distance_matrix, args, dict = cli.run()

score = -1
if np.amax(clusters) > 0:
    score = silhouette_score(distance_matrix, clusters, metric='precomputed')

book = None
if not os.path.isfile("outputs/bookkeeping.csv"):
    book = pd.DataFrame(columns=['matrix_name', 'clusters', 'silhouette_score', 'method', 'metric', 'epsilon', 'neighbours'])
else:
    book = pd.read_csv("outputs/bookkeeping.csv", index_col=0)
row = {'matrix_name': args.matrix, 'clusters': np.amax(clusters) + 1, 'silhouette_score': score}
row.update(dict)
book = book.append(row, ignore_index=True)
book.to_csv("outputs/bookkeeping.csv")

df = pd.read_csv(args.dataset, index_col=0, delimiter='\t')
df['cluster'] = clusters
df.to_csv("outputs/library_clustering.csv")


