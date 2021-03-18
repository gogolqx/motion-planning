import argparse
import shutil
import pandas as pd
import numpy as np
import os

parser = argparse.ArgumentParser(description='A simple moving tool')
parser.add_argument('dataset', type=str, help='A csv file path')
args = parser.parse_args()

if os.path.isdir('outputs/animations'):
    shutil.rmtree('outputs/animations')
df = pd.read_csv(args.dataset)
num_clusters = np.amax(df['cluster']) + 1
for i in range(num_clusters):
    tmp = df[df['cluster'] == i]
    num_observations = min(10, tmp.shape[0])
    tmp = tmp[:num_observations]
    os.makedirs('outputs/animations/cluster_' + str(i))
    for index, row in tmp.iterrows():
        source = 'data/gif/' + row['scenario_name'] + '.gif'
        dest = 'outputs/animations/cluster_' + str(i) + '/' + row['scenario_name'] + '.gif'
        if os.path.isfile(source):
            shutil.copy(source, dest)
