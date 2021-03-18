"""Simple inteface for RF and proximity matrix algorithm."""
import argparse
import os
import pickle
import sys
import time

import cluster.common as cm
import cluster.decision_tree as dt
import cluster.proximity as px

import matplotlib.pyplot as plt

import pandas as pd

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import TruncatedSVD


def generate_df(file, delimiter='\t', col=None):
    """Generating Dataframe from a csv file.

    Args:
        col: a list of strings which includes column names that chosen
        to generate the dataframe. Default is None(All columns are selected).
    Returns:
        A dataframe
    """
    df = cm.read_csv(file, delimiter=delimiter)
    df = cm.rename(df)
    if col is not None:
        df = cut_df(df, col)
    df['label'] = 1
    if df is not None:
        return df
    else:
        return None


def reduce_df(df, dimensions=10, save=False):
    """."""
    print("Running SVD")
    svd = TruncatedSVD(n_components=dimensions, n_iter=10)
    ret = svd.fit_transform(df.drop(['scenario_id', 'scenario_name'], axis=1))
    ret = pd.DataFrame(ret)
    ret['scenario_name'] = df['scenario_name']
    ret['scenario_id'] = df['scenario_id']
    if save:
        ret.to_csv("data/reduced.csv")
    return ret


def cut_df(df, col):
    """."""
    if set(col).difference(set(df.columns)) == set():
        df = pd.DataFrame(df, columns=dt.NO_FEATURES + col)
        return df
    else:
        print('Given column names are not in the dataframe.\
               Please check again.')
        return None


def create_d_matrix(rf_results, n_rows):
    """."""
    p_matrix = px.avg_matrix(rf_results, n_rows)
    # the higher the p_scores the smaller the distance between pairs
    d_matrix = 1 - p_matrix
    for i in range(len(d_matrix)):
        d_matrix[i][i] = 0
    return d_matrix


def hierarch_cluster(matrix):
    """."""
    fig = plt.figure(figsize=(10, 10))
    z = linkage(matrix, method='ward', metric='euclidean')
    dendrogram(z, truncate_mode='lastp', p=30,
               show_leaf_counts=True, leaf_rotation=90,
               leaf_font_size=15, show_contracted=True)
    plt.title('Dendrogram for the Agglomerative Clustering')
    fig.savefig("outputs/hierarch_tree.pdf")
    return z


def classify(df, z, num_clusters=8):
    """."""
    fig, axes = plt.subplots(nrows=num_clusters,
                             ncols=1, figsize=(10, 50))
    plt.subplots_adjust(wspace=0, hspace=3)
    labels = fcluster(z, t=num_clusters, criterion='maxclust')
    labeled_dic = {}
    for i, label in enumerate(labels):
        if label not in labeled_dic:
            labeled_dic[label] = [i]
        else:
            labeled_dic[label].append(i)
    # total_cols = len(df.columns)
    # for key in labeled_dic:
        # df_cluster = (df[df[SCENARIO_ID].isin(labeled_dic[key])])
        # df_cluster.ix[:, range(3, total_cols)].boxplot(ax=axes[key - 1],
            # rot=90)
    # fig.savefig("outputs/cluster_boxplot.pdf", bbox_inches='tight')
    for key in labeled_dic:
        df.loc[df['scenario_id'].isin(labeled_dic[key]),
               'classification'] = str(key)
    return df, labeled_dic


def run_rf(df, num_t, terminal_leaves, min_gini, max_gini, detailed=False):
    """."""
    rf_result, stats_list = dt.random_f(df,
                                        frac=0.5,
                                        terminal_leaves=terminal_leaves,
                                        min_gini=min_gini,
                                        max_gini=max_gini,
                                        num_t=num_t,
                                        detailed=detailed)
    summary = {}
    for col in df.columns:
        summary[col] = 0
    for stat in stats_list:
        for key in stat:
            summary[key] += stat[key]
    # print("summary: \n", summary)
    return rf_result, summary


parser = argparse.ArgumentParser(description="Scenarios clustering!")
parser.add_argument('-d', '--dataset', required=True, type=str, dest='dataset',
                    help='A csv file path. (columns must contain {}.)\
                    '.format(dt.SCENARIO_ID))
parser.add_argument('-f', '--features', nargs='+',
                    dest='features', help='Selected features must be included \
                    in given dataset\'s columns.')
parser.add_argument('-nt', '--num_trees', default=1, type=int,
                    dest='num_trees', help='Number of trees in RF algorithm.')
parser.add_argument('-tl', '--terminal_leaves', default=100, type=int,
                    dest='terminal_leaves', help='Minimum number of leaves to stop \
                    splitting in RF algorithm.')
parser.add_argument('-ming', '--min_gini', default=0.05, type=float,
                    dest='min_gini', help='Minimum gini value for pruning tree \
                    in RF algorithm.')
parser.add_argument('-maxg', '--max_gini', default=0.35, type=float,
                    dest='max_gini', help='Minimum gini value for pruning tree \
                    in RF algorithm.')
parser.add_argument('-nc', '--num_clusters', default=8, type=int,
                    dest='num_clusters', help='number of clusters for\
                    classification.')
parser.add_argument('-l', '--len_data', default=100000, type=int,
                    dest='len_data', help='chosen length of dataset')
parser.add_argument('--debug', action='store_true',
                    dest='debug', help='Show detailed output')
parser.add_argument('-r', '--reduce', type=int,
                    dest='reduce', help='Perform SVD on dataset. Set number of \
                     dimensions')

args = parser.parse_args()
SUFFIX = time.time()
print("SUFFIX: ", SUFFIX)
if not args.dataset:
    print(parser.parse_args(['-h']))
    sys.exit()
else:
    df = generate_df(args.dataset, col=args.features, delimiter='\t')
    if args.reduce is not None:
        print("Reducing dimensionality to " + str(args.reduce))
        df = reduce_df(df, args.reduce, True)
    if len(df) > args.len_data:
        df = df[:args.len_data]
    print('length of df is: ', len(df))
    print('1. running random forest...')
    rf_result, summary = run_rf(df, num_t=args.num_trees,
                                min_gini=args.min_gini,
                                max_gini=args.max_gini,
                                terminal_leaves=args.terminal_leaves,
                                detailed=args.debug)

    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    pickle.dump(rf_result, open("outputs/test_result_{}.pkl".format(SUFFIX),
                                'wb'))
    cm.dict_to_csv(summary, 'outputs/summary_{}.csv'.format(SUFFIX))

    print('2. creating proximity(distance) matrix...\
          please have more patience to wait...')
    n_rows = len(df)
    d_matrix = create_d_matrix(rf_result, n_rows)
    pickle.dump(d_matrix, open("outputs/matrix_{}.pkl".format(SUFFIX),
                               "wb"))
    # pd.DataFrame(d_matrix).to_csv("outputs/matrix_{}.csv".format(SUFFIX))

    # print('3. hierachical clustering...')
    # z = hierarch_cluster(d_matrix)
    # df, labeled_dic = classify(df, z, num_clusters=args.num_clusters)
    # df.to_csv("outputs/test_final_{}.csv".format(SUFFIX))
    print('Distance matrix generated!')
