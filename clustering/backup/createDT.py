import algs.decision_tree as dt
import os
import numpy as np 
import pandas as pd
print('ddddddddd')
print(os.getcwd())
BASIC_PATH = os.getcwd()+'/highD-dataset/Python'
recording_ids = [str(idx) if not len(str(idx)) == 1 else '0' + str(idx) for idx in list(range(1,61))]
data_prefix = BASIC_PATH+'/data/'
image_suffix = '_highway.jpg'
recording_meta_suffix = '_recordingMeta.csv'
tracks_meta_suffix = '_tracksMeta.csv'
tracks_suffix = '_tracks.csv'

track_meta_01 = pd.read_csv(data_prefix + recording_ids[0] + tracks_meta_suffix).set_index('id')
filtered_track_meta_01 = track_meta_01[(track_meta_01['numFrames'] >= 100) & (track_meta_01['minDHW'] > 0) & (track_meta_01['minTHW'] > 0) & (track_meta_01['minTTC'] > 0) & (track_meta_01['numLaneChanges'] > 0)]
filtered_track_meta_01.head()
df = filtered_track_meta_01
df['label'] = 1
df = df.reset_index()

df = df.drop('initialFrame',axis =1)
df = df.drop('finalFrame',axis =1)
df = df.drop('numFrames',axis =1)
df = df[df.drivingDirection == 1]
df = df.drop('drivingDirection',axis =1)
#df = df[df['class']==1]
df = df.drop('class', axis=1)
df = df.drop('traveledDistance', axis=1)
df = df.drop('width', axis=1)
df = df.drop('height', axis=1)

if __name__ == "__main__":

	test_tree_10 = dt.build_tree(df[:10],5)
	test_dict=dt.pre_traversal(test_tree_10,0)

	