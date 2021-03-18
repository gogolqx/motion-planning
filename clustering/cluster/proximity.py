"""Modul for Proximity Matrix Algorithm."""
import numpy as np


def compute_p_score(str1, str2):
    """Compare two paths, return the size of intersect set.

    example: str1 = LLRRLLR
             str2 = LLLR

             same = |LL| = 2
    """
    maxlen = len(str2) if len(str1) < len(str2) else len(str1)
    same = 1
    # loop through the characters
    for i in range(maxlen):
        if str1[i:i + 1] == str2[i:i + 1]:
            same += 1
        else:
            break
    denominator = 2 + len(str1) + len(str1) - same
    p_score = same / denominator
    return p_score


def avg_matrix(rf_results, n_rows):
    """Generate the final proximity matrix given RF_results."""
    size = n_rows
    matrix_sum = np.zeros((size, size))
    count_m = np.zeros((size, size))
    for index, tree in enumerate(rf_results):
        treelist = tuple(tree.items())
        print('generating matrix for tree ', index)
        matrix, subcount_m = generate_score_matrix(treelist, n_rows)
        matrix_sum += matrix
        count_m += subcount_m
    # averaging scores from all trees
    matrix_sum = matrix_sum / count_m
    matrix_sum = matrix_sum + matrix_sum.T
    # set diagnal value 1
    matrix_sum[np.diag_indices_from(matrix_sum)] = 1
    return matrix_sum


def generate_score_matrix(treelist, n_rows):
    """Generate single P_score matrix and related count_matrix for one tree."""
    size = n_rows
    matrix = np.zeros((size, size))
    count_matrix = np.ones((size, size))
    for i, x in enumerate(treelist):
        # x looks like this:((1, 'R'), (4, 'R'),...)
        for y in treelist[i:]:
            # build up a dictionary for store the scores
            # calculate score of pair scenarios
            matrix[x[0]][y[0]] = compute_p_score(x[1], y[1])
            count_matrix[x[0]][y[0]] += 1
    return matrix, count_matrix


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """Check whether matrix is symmetric."""
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
