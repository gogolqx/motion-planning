"""Modul for modified Unsupervised Random Forest Algorithm."""
import random

import numpy as np

import pandas as pd

SCENARIO_ID = 'scenario_id'
# EGO_ID = 'ego_id'
LABEL = 'label'
SCENARIO_NAME = 'scenario_name'
NO_FEATURES = [SCENARIO_ID, LABEL, SCENARIO_NAME]
# helping for float values to calculate best splitting value
LINSPACE = 50
verbose = True
MAX_GINI = 0.35


class DecisionNode:
    """A Class of decision node which is a element of binary tree.

    Attributes:
        col: a string. the name of splitting criterion feature .

        value: a float number for continous feature;
        a integer for discrete feature.
        splitting value of this criterion. (choose where to split)

        right_b:a DecisionNode.  right branch of this node

        left_b: a DecisionNode. left branch of this node

        results: a dictionary. if it is a end node
        (leaf, has neither right_b nor left_b).
        it contains all classified scenarios.

        depth: the root node itself is on depth 0.
    """

    def __init__(self, col=-1, value=None, results=None,
                 left_b=None, right_b=None, depth=0):
        """."""
        self.col = col    # col is the name of feature
        self.value = value  # value is corresponding the best split value
        self.left_b = left_b  # left branch
        self.right_b = right_b  # right branch
        self.results = results  # save all scenarios with id
        self.depth = depth


def separate_cols(df):
    """Sort columns into float and int given dataset."""
    df = check_drop(df)[0]
    if verbose:
        print('after drop: ', df.columns)
    float_cols = []
    int_cols = []
    for col in df.columns:
        if 'float64' == df.dtypes[col]:
            float_cols.append(col)
        elif 'int64' == df.dtypes[col]:
            int_cols.append(col)
    return float_cols, int_cols


def check_drop(df, cols=NO_FEATURES):
    """Drop the cols if it exists."""
    drop = 0
    for col in cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
            drop += 1
    return df, drop


def build_tree(df, min_nodes, float_cols, int_cols, stat, min_gini, max_gini,
               deep=0):
    """
    From one dataframe (with feature columns) to generate a tree.

    options: min_nodes:= # the minimum number of real datapoints
    in order to make a new split
    """
    if len(df) == 0:
        return DecisionNode()
        # if the size of tree is bigger than the minimum nodes
    if len(df) > min_nodes:
        best_feature_list = []
        best_f = None
        t_v = None
        # separate the features into discrete and continuous
        # compute the range for the uniform distributed synthetic dataset
        min_max = calculate_min_max(df[float_cols])
        # choose the feature with minimum gini impurity
        best_feature_list = select_localbest(df, min_max, float_cols, int_cols)
        # get the name of feature and the corresponding split value
        best_f, t_v, best_gini = global_best_feature(best_feature_list)
        if verbose:
            print("in depth {} : the best_gini, best_f:".format(deep))
            print(best_gini, best_f)
        if best_gini >= min_gini and best_gini < max_gini:
            # split the data based on best_f
            df_set1, df_set2 = split(df, best_f, t_v)
            # recursion
            stat[best_f] += 1
            branch1 = build_tree(df_set1, min_nodes=min_nodes,
                                 float_cols=float_cols,
                                 int_cols=int_cols,
                                 deep=deep + 1,
                                 min_gini=min_gini,
                                 max_gini=max_gini,
                                 stat=stat)[0]
            branch2 = build_tree(df_set2, min_nodes=min_nodes,
                                 float_cols=float_cols,
                                 int_cols=int_cols,
                                 deep=deep + 1,
                                 min_gini=min_gini,
                                 max_gini=max_gini,
                                 stat=stat)[0]
            return DecisionNode(col=best_f, value=t_v,
                                left_b=branch1,
                                right_b=branch2,
                                depth=deep), stat
        else:
            if verbose:
                print("Gini value does not satisfy! Splitting stopped.")
            return DecisionNode(results=df[SCENARIO_ID].tolist(),
                                depth=deep), stat
    else:
        return DecisionNode(results=df[SCENARIO_ID].tolist(),
                            depth=deep), stat


def calculate_min_max(df):
    """Compute the range (maximum/minimum value) of each continuous feature."""
    a = df.min()
    b = df.max()
    min_max = pd.concat([a, b], axis=1)
    min_max.columns = ['min', 'max']
    return min_max.T


def extract_unique(df, df_int):
    """Extract the unique values from discrete features."""
    int_features_value_dic = {}
    for f in df_int:
        int_features_value_dic[f] = df[f].unique()
    return int_features_value_dic


def create_fake_intlist(df, int_dic, col):
    """
    Create synthetic data list.

    for a split which has a discrete feature
    criterion.
    """
    values = int_dic[col]
    df_fake_list = values.repeat(len(df) / len(values))

    return df_fake_list


def create_fake_floatlist(df, min_max, col):
    """
    Create synthetic data list.

    for a split which has a continuous feature criterion.

    """
    ran = min_max[col].tolist()
    df_fake_list = np.linspace(ran[0], ran[1], num=len(df)).tolist()
    return df_fake_list


def calculate_gini(col, t_list, df_total):
    """
    Given a col (feature), a list of splitting values (t_list).

    Calculate gini impurity on each values
    """
    gini_list = []
    for tv in t_list:
        df1 = df_total[df_total[col] <= tv]
        df2 = df_total[df_total[col] > tv]
        p1 = len(df1[df1.label == 1]) / len(df1)
        gini1 = 2 * p1 * (1 - p1)
        if len(df2) != 0:
            p2 = len(df2[df2.label == 1]) / len(df2)
            gini2 = 2 * p2 * (1 - p2)
            gini = (gini1 * len(df1) / len(df_total) +
                    gini2 * len(df2) / len(df_total))
        else:
            p2 = 0
            gini = gini1 * len(df1) / len(df_total)
        gini_list.append([gini, tv])
    gini_list = np.asarray(gini_list)

    return gini_list


def synthetic_datasets(df_list, df_fake_list, col):
    """
    Given the split criterion.

    add the corresponding ensemble noise into real dataset
    """
    df_fake = pd.DataFrame(df_fake_list)
    df_fake['label'] = 0
    df_original = pd.DataFrame(df_list)
    df_original['label'] = 1
    df_total = pd.concat([df_original, df_fake])
    df_total.columns = [col, 'label']

    # def_total looks like this:
    #     l_rel_pos_end  label
    # 0       13.457300      1
    # 1       13.341600      1
    # 2       23.220200      1
    # 0       13.457300      0
    # 1       18.043200      0
    # 2       23.220200      0
    total_list = np.concatenate((df_list, df_fake_list))
    total_list = sorted(total_list)
    return total_list, df_total


def calculate_best_split(df_list, df_fake_list, col,
                         discrete=False, discrete_dic=None):
    """Compute the best value for one feature to split."""
    x, df_total = synthetic_datasets(df_list, df_fake_list, col)
    maxi = max(x)
    mini = min(x)
    if not discrete:
        if len(x) < LINSPACE:
            t_list = np.linspace(maxi, mini, num=len(x))
        else:
            t_list = np.linspace(maxi, mini, num=LINSPACE)
    else:
        t_list = discrete_dic[col]

    gini_list = calculate_gini(col, t_list, df_total)
    # gini_list looks like:
    # array([[0.5 , 1.  ],
    #    [0.4 , 0.8 ],
    #    [0.36, 0.6 ]])
    min_index = gini_list.argmin(0)[0]
    best_t = gini_list[min_index]
    return (best_t[0], best_t[1], col)


def select_localbest(df, min_max, df_float, df_int):
    """
    Select the minimum gini impurity of each feature.

    return a list which contains the name of this feature,
    the gini impurity value and its best split value.

    """
    best_feature_list = []
    # feature select :
    for icol in min_max[df_float].columns:
        df_fake_list = create_fake_floatlist(df, min_max, icol)
        df_list = df[icol].tolist()
        best_t = calculate_best_split(df_list, df_fake_list, icol)
        best_feature_list.append(best_t)
    if df_int != []:
        int_dic = extract_unique(df, df_int)
        for icol in df_int:
            df_fake_list = create_fake_intlist(df, int_dic, icol)
            df_list = df[icol].tolist()
            best_t = calculate_best_split(df_list, df_fake_list, icol,
                                          discrete=True, discrete_dic=int_dic)
            best_feature_list.append(best_t)
    return best_feature_list


def global_best_feature(best_feature_list):
    """
    Select the global minimum gini impurity from all features.

    return a list which contains the name of this feature
    and its best split value

    """
    x = pd.DataFrame(best_feature_list, columns=['gini', 't_value', 'feature'])
    best_gini = x['gini'].min()
    best_row = x[x['gini'] == best_gini]
    best_feature = None
    t_value = None
    for index, row in best_row.iterrows():
        best_feature = row['feature']
        t_value = row['t_value']
    if verbose:
        print("best_feature: ", best_feature)
        print("best_gini: ", best_gini)
    return best_feature, t_value, best_gini


def split(df, feature, t_value):
    """
    Given the splitting criterion and splitting value to split.

    return two dataframes.
    traversal the tree from the root node to end node
    return a dictionary whose value is the clustered scenarios (as end node,
    which contains the scenario ids)
    whose key is the path of a node.
    Example:
    {
    '0L1L2L3R': {13: 123, 16: 139, 44: 468, 84: 860, 103: 1029},
    '0L1R2L': {62: 639, 66: 665, 71: 726},
    '0L1R2R': {21: 161, 82: 835, 90: 926},
    '0R1L': {10: 116, 47: 475, 75: 745, 91: 928},
    '0R1R': {20: 160, 54: 531}
    }
    the path means a set including all nodes the datapoint passed,
    namely from root node to itself.
    Format example : '0L1L2L3R' represents the node path is
    from the root node (0-depth),
    goes three consecutive times to the left branch and
    terminates to the right.
    """
    # left branch are smaller than or equal to t
    df_set1 = df[df[feature] <= t_value]
    # right branch are greater than t
    df_set2 = df[df[feature] > t_value]
    return df_set1, df_set2


def pre_traversal(tree, depth=0, branch=None, pos="", node_dict={}):
    """."""
    # it has branches
    if tree.value is not None:
        cur_pos_l = pos + str(depth) + "L"
        cur_pos_r = pos + str(depth) + "R"
        if tree.left_b.results is not None:
            node_dict[cur_pos_l] = (tree.left_b.results)
        if tree.right_b.results is not None:
            node_dict[cur_pos_r] = (tree.right_b.results)
        pre_traversal(tree.left_b, depth + 1, branch='L',
                      pos=pos + str(depth) + 'L',
                      node_dict=node_dict)
        pre_traversal(tree.right_b, depth + 1, branch='R',
                      pos=pos + str(depth) + 'R',
                      node_dict=node_dict)
    return node_dict


def pre_traversal_nodepth(tree, depth=0, branch=None, pos="", node_dict={}):
    """Traversal the whole tree and return a dictionary.

    example of return value:
    {'R': [1,4,7],'LR': [2], 'LL: [3,5,8]'}
    """
    if tree.value is not None:
        # if it has branches:
        cur_pos_l = pos + "L"
        cur_pos_r = pos + "R"
        if tree.left_b.results is not None:
            node_dict[cur_pos_l] = tree.left_b.results
        if tree.right_b.results is not None:
            node_dict[cur_pos_r] = (tree.right_b.results)
        pre_traversal_nodepth(tree.left_b, depth + 1, branch='L',
                              pos=pos + 'L',
                              node_dict=node_dict)
        pre_traversal_nodepth(tree.right_b, depth + 1, branch='R',
                              pos=pos + 'R',
                              node_dict=node_dict)
    else:
        node_dict[pos] = tree.results
    return node_dict


def reverse_key_value(dic):
    """Exchange the key and value of a dictionary.

    example: {'R': [1,4,7],'LR': [2], 'LL': [3,5,8]'} would be reversed into
    {1:'R',4:'R',7:'R',2:'LR',3:'LL',5:'LL',8:'LL'}
    """
    inv_map = {}
    for k, v in dic.items():
        for item in v:
            inv_map[item] = k
    return inv_map


def random_f(df, min_gini, max_gini, num_t=10, terminal_leaves=5,
             frac=0.5, nodepth=True, detailed=False):
    """Build random forest based on a dataset.

    Attributes:
            df: dataFrame.

            num_t: number of trees;

            min_gini: minimum gini value for stopping splitting the branch.

            terminal_leaves: minimum number of leaves
            for stopping splitting the branch.

            frac: bagging ratio. between 0 and 1.

            nodepth: boolean, choosing travarsal mode (show depth or not).

            detailed: show splitting and feature selection output for each node
    return: A list consists of
    """
    global verbose
    verbose = detailed

    total_results = []
    float_cols, int_cols = separate_cols(df)
    if verbose:
        print("float_cols: ", float_cols)
        print("int_cols: ", int_cols)
    df_f = check_drop(df)[0]
    num_fea = int(np.sqrt(len(df_f.columns)))
    summary = []
    for i in range(num_t):
        dic_b = {}
        result_b = {}
        tree_b = None
        print('Generating Tree ', i + 1, '..')

        chosen_feas = random.sample(df_f.columns.tolist(),
                                    num_fea)
        if verbose:
            print("chosen_feas: ", chosen_feas)
        chosen_float_cols = []
        chosen_int_cols = []
        stat = {}
        for feature in chosen_feas:
            stat[feature] = 0
            if feature in float_cols:
                chosen_float_cols.append(feature)
            else:
                chosen_int_cols.append(feature)
        # Bagging for choosing scenarios to generate trees
        df_b = df.sample(frac=frac, replace=False)

        tree_b, stat = build_tree(df_b,
                                  stat=stat,
                                  float_cols=chosen_float_cols,
                                  int_cols=chosen_int_cols,
                                  min_nodes=terminal_leaves,
                                  min_gini=min_gini,
                                  max_gini=max_gini,
                                  deep=0)
        summary.append(stat)
        if nodepth:
            dic_b = pre_traversal_nodepth(tree_b, node_dict={})
        else:
            dic_b = pre_traversal(tree_b, node_dict={})
        result_b = reverse_key_value(dic_b)
        total_results.append(result_b)
    return total_results, summary
