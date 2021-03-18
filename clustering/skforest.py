import argparse
import sys
import time

import numpy as np
import pickle

import cluster.decision_tree as dt
import cluster.common as cm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def generate_df(file, delimiter=',', col=None):
    """Generating Dataframe from a csv file.

    Args:
        col: a list of strings which includes column names that chosen
        to generate the dataframe. Default is None(All columns are selected).
    Returns:
        A dataframe
    """
    df = cm.read_csv(file, delimiter)
    if col is not None:
        df = cut_df(df, col)
    df['label'] = 1
    if df is not None:
        return df
    else:
        return None

def cut_df(df, col):
    """."""
    if set(col).difference(set(df.columns)) == set():
        df = pd.DataFrame(df, columns=dt.NO_FEATURES + col)
        return df
    else:
        print('Given column names are not in the dataframe.\
               Please check again.')
        return None


def generate_synthetic(df, num_synthetic):
    floats, ints = dt.separate_cols(df)
    bounds = {}
    ret = df.copy()
    # Making sure label is not part of list
    if 'label' in ints:
        ints.remove('label')
    if 'timespan' in floats:
        floats.remove('timespan')
    for col in ret.columns:
        bounds[col] = (np.amin(ret[col]), np.amax(ret[col]))
    for _ in range(num_synthetic):
        row = {'label': 2, 'timespan': 3.2}
        for c in ints:
            row[c] = np.random.randint(bounds[c][0], bounds[c][1])
        for c in floats:
            row[c] = np.random.uniform(bounds[c][0], bounds[c][1])
        ret = ret.append(row, ignore_index=True)
    #ret.to_csv("data/synthetic.csv")
    return ret


def proximity_matrix(model, X, normalize=True):

    terminals = model.apply(X)
    n_trees = terminals.shape[1]

    a = terminals[:,0]
    prox_mat = 1*np.equal.outer(a, a)

    for i in range(1, n_trees):
        a = terminals[:,i]
        prox_mat += 1*np.equal.outer(a, a)

    if normalize:
        prox_mat = prox_mat / n_trees

    return prox_mat


def fit_forest(df, num_trees=300, gini=0.1, leafs=None):
    rfc = RandomForestClassifier(n_estimators=num_trees, min_impurity_split=gini, n_jobs=-1, max_leaf_nodes=leafs)
    num_real = df[df['label'] == 1].shape[0]
    print(num_real)
    X = df.loc[:, (df.isnull().sum(axis=0) <= 0)]
    y = X['label']
    X = X.drop(['label'], axis=1)
    rfc.fit(X, y)
    dist_matrix = 1 - proximity_matrix(rfc, X, normalize=True)
    return dist_matrix[:num_real, :num_real]

parser = argparse.ArgumentParser(description="Scikit learn random forest")
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
parser.add_argument('-g', '--min_gini', default=0.3, type=float,
                    dest='min_gini', help='Minimum gini value for pruning tree \
                    in RF algorithm.')
parser.add_argument('-s', '--num_synthetic', default=20000, type=int,
                    dest='synthetic', help='Number of synthetic data points.')

SUFFIX = time.time()
print("SUFFIX: ", SUFFIX)
args = parser.parse_args()
if not args.dataset:
    print(parser.parse_args(['-h']))
    sys.exit()
else:
    df = generate_df(args.dataset, col=args.features)
    df = generate_synthetic(df, args.synthetic)
    d_matrix = fit_forest(df)
    pickle.dump(d_matrix, open("outputs/lmatrix_{}.pkl".format(SUFFIX),
                               "wb"))
    pd.DataFrame(d_matrix).to_csv("outputs/lmatrix_{}.csv".format(SUFFIX))
