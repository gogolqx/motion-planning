# Generating CommonRoad Scenarios from highD dataset
In order to generate CommonRoad Scenarios from highD dataset, we need to use the
`highd_to_cr.py` script under the `data_generation/highd_generator` folder.

We use the following command:

``
python highd_to_cr.py <path to highd dataset's 'data' folder's root> <output path> --num_timesteps 40 --metric_id <metric id>
``

Metric IDs:
 - 0: Use DHW metric
 - 1: Use THW metric
 - 2: Use TTC metric
 - 3 or bigger: Use all metrics and generate datasets for each of them

**Note:** To include only the lane changing tracks, append `--lc` to the command.


# Extracting features from CommonRoad Scenario Files
In order to extract features from generated CommonRoad Scenarios, we need to use
the `feature_extraction.py` script under the `feature_extraction` folder.

We use the following command:

``
python feature_extraction.py <path to scenario file or folder> --out <output path> --f --ego --ego-id 99
``

Meaning of the arguments:
 - out: If an output was not specified, it will output a feature.csv file in the same folder.
 - f: This is a flag to indicate the given path is to a folder containing the scenario files, if not specified, it will asume the path is for a single file
 - egp: This is a flag to indicate the ego vehicle was given in the scenario as a dynamic obstacle
 - ego-id: This is the obstacle id of the ego vehicle


# Running Random Forest and Clustering using Docker

In order to run clustering on datasets automatically, `docker` and `docker-compose`
needs to be installed.

Afterwards, there are two options to run clustering on datasets:
 - Running a single clustering on a single dataset using `start_clustering.sh`
 - Running 24 clustering back to back on all datasets using `run_all_clusters.sh`

In both cases, if there are changes to the code (in clustering subfolder), dataset
csv files, or to the environment files for the parameters, they will all reflect
in the created docker containers, so clustering will be up to date.


##### Running all clusterings on all datasets back to back.

In order to run clusterings on datasets back to back by batches of 4, run the
following command on the root of the repository:

``
bash run_all_clusters.sh
``

This command will start clustering algorithms. The batches are:
 - Clustering using all features of a dataset
 - Clustering using dhw features of a dataset
 - Clustering using thw features of a dataset
 - Clustering using ttc features of a dataset

Since there are 6 different datasets, there will be 6 different batches running
back to back.

The results of clustering will be placed to `results` folder in the root of the
repository with the individual clustering's service name as subfolder. (For service
names see next section.)

##### Running a single clustering on a single dataset

In order to run single clustering on a dataset we need to run the following command
on the root of the repository:

``
bash start_clustering.sh <service-name>
``

Service names are defined in the `docker-compose.yaml` file. The options are:
 - **dhw-small-all**  ->  Clustering using all features of dhw-onlylanechanging dataset
 - **dhw-small-dhw**  ->  Clustering using dhw features of dhw-onlylanechanging dataset
 - **dhw-small-thw**  ->  Clustering using thw features of dhw-onlylanechanging dataset
 - **dhw-small-ttc**  ->  Clustering using ttc features of dhw-onlylanechanging dataset
 - **thw-small-all**  ->  Clustering using all features of thw-onlylanechanging dataset
 - **thw-small-dhw**  ->  Clustering using dhw features of thw-onlylanechanging dataset
 - **thw-small-thw**  ->  Clustering using thw features of thw-onlylanechanging dataset
 - **thw-small-ttc**  ->  Clustering using ttc features of thw-onlylanechanging dataset
 - **ttc-small-all**  ->  Clustering using all features of ttc-onlylanechanging dataset
 - **ttc-small-dhw**  ->  Clustering using dhw features of ttc-onlylanechanging dataset
 - **ttc-small-thw**  ->  Clustering using thw features of ttc-onlylanechanging dataset
 - **ttc-small-ttc**  ->  Clustering using ttc features of ttc-onlylanechanging dataset
 - **dhw-big-all**  ->  Clustering using all features of dhw dataset
 - **dhw-big-dhw**  ->  Clustering using dhw features of dhw dataset
 - **dhw-big-thw**  ->  Clustering using thw features of dhw dataset
 - **dhw-big-ttc**  ->  Clustering using ttc features of dhw dataset
 - **thw-big-all**  ->  Clustering using all features of thw dataset
 - **thw-big-dhw**  ->  Clustering using dhw features of thw dataset
 - **thw-big-thw**  ->  Clustering using thw features of thw dataset
 - **thw-big-ttc**  ->  Clustering using ttc features of thw dataset
 - **ttc-big-all**  ->  Clustering using all features of ttc dataset
 - **ttc-big-dhw**  ->  Clustering using dhw features of ttc dataset
 - **ttc-big-thw**  ->  Clustering using thw features of ttc dataset
 - **ttc-big-ttc**  ->  Clustering using ttc features of ttc dataset

Example:
``
bash start_clustering.sh dhw-small-all
``

##### Changing Random Forest and Clustering Parameters

In order to change the parameters of the random forest and clustering scripts,
only the `<featurecombination>.env` files need to be modified.

List of **.env** files:
 - all.env  -> when running clustering using all features
 - dhw.env  -> when running clustering using dhw features
 - thw.env  -> when running clustering using thw features
 - ttc.env  -> when running clustering using ttc features

# scenario-clustering

Clustering Similar Traffic Scenarios - Motion planning for autonomous vehicles praktikum

## Random Forest and Proximity Implementation: Interface one

This is a QiaoXi's branch --- implementing random forest algorithm as well as path proximity scores.

### Core codes are in the folder /codes
 - decision_tree.py (functions for random forest and decision tree)
 - proximity.py (functions for computing path proximity scores)

### Interfaces with other team partners is

- run.py
 
#### Test/run the RF algorithm:
- First make sure you already installed all packages. If not, please run 
``
pip install -r requirements.txt
``
- Before you run, please have a look for the **notes** below which mention the format of input file.
- Then run this in Terminal: 
```
run.py -d <csv file path>
```
or add `-h` to see more details.

Example: (generating 10 trees and terminal leaves is 100.)

```
python run.py -d data/featuresid.csv -nt 10 -tl 50
```

### Some Notes:
1. Input

The standard input file should 
- be **.csv**; 
- include one column named **"scenario_id"** which is unique;
- type of "scenario_id" column should be **integer**.
- Recommand to use directly the feature name as column name.

2. Outputs

In terms of my task, you could get a summary from running random forest algorithm as well as a proximity matrix. Furthermore you can also get a classification result from this interface one. To do more explicity work on classifying please continue to use the interface from Emanuel. 

There are now following outputs:
- **summary.csv** : The frequencies of each feature (shown as the column name) used for splitting during the whole random forest algorithm (sum from all trees)
- **matrix.pkl** : A pickle file which storage the proximity matrix (The type is numpy.matrix). If the size of dataset is more than 1000, you need **patience** to wait while generating the huge matrix.
- hierarch_tree.pdf : The result from hierachical clustering. 
- test_final_c.csv : The final result from classification

The last two one could be changed in terms of Emanuel's work. (He takes charge of the second step after generating the proximity (distance) matrix.)

3. Console

You can also have a look at console when it is running. It shows some details from the processing, such as the chosen features, best global feature and gini value at each split, as well as a statistic at the end of generating one tree. It collects the frequencies of splitting features in generating this tree.

Example:
```
Generating Tree  10 ..
chosen_feas:  ['ll_rel_pos_init', 'pr_rel_pos_mid', 'ego_v_init', 'l_rel_pos_mid', 'p_rel_pos_init']
chosen_float_cols:  ['ll_rel_pos_init', 'pr_rel_pos_mid', 'ego_v_init', 'l_rel_pos_mid', 'p_rel_pos_init']
best_feature_list:  [(0.3359893333333333, -1.0, 'll_rel_pos_init'), (0.40142799908424914, -1.0, 'pr_rel_pos_mid'), (0.43456520921839165, 23.934164228456915, 'ego_v_init'), (0.3631596236420376, -1.0, 'l_rel_pos_mid'), (0.4346240459156221, 56.871720941883765, 'p_rel_pos_init')]
best_feature:  ll_rel_pos_init
best_gini:  0.3359893333333333
best_feature_list:  [(0.5, -1.0, 'll_rel_pos_init'), (0.42410329985652795, 46.67592943548387, 'pr_rel_pos_mid'), (0.43852240989493285, 24.132267943548385, 'ego_v_init'), (0.27134932561548025, -1.0, 'l_rel_pos_mid'), (0.4398243392941921, 56.71674495967742, 'p_rel_pos_init')]
best_feature:  lr_rel_pos_end
best_gini:  0.44285714285714284
best_gini bigger than min_gini! Splitting stopped.
stat:
 {'ego_breaktime_until_mid': 6, 'lr_rel_pos_end': 3, 'l_rel_pos_end': 4, 'p_rel_pos_end': 1, 'surr_veh_count_end': 0}
 ```
