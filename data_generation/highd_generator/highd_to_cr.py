import os
import glob
import copy
import time
import argparse
import pandas as pd

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile

from map_utils import *
from obstacle_utils import *
from planning_utils import *

AUTHOR = 'Xiao Wang'
AFFILIATION = 'Technical University of Munich, Germany'
SOURCE = 'The HighWay Drone Dataset (highD)'
TAGS = 'highway multi_lane parallel_lanes no_oncoming_traffic'

# dicts to map location id to location names and obstacle types
location_dict = {
    1: "LocationA",
    2: "LocationB",
    3: "LocationC",
    4: "LocationD",
    5: "LocationE",
    6: "LocationF"
}


def get_file_lists(path):
    listing = glob.glob(path)
    listing.sort()
    return listing


def get_args():
    parser = argparse.ArgumentParser(description="Generates CommonRoad scenarios from highD dataset")
    parser.add_argument('highd_dir', metavar='h', type=str, help='Path to highD data files')
    parser.add_argument('output_dir', metavar='o', type=str, help='Directory to store generated .xml files')
    parser.add_argument('--num_timesteps', type=int, default=40)
    parser.add_argument('--num_planning_problems', type=int, default=1)
    parser.add_argument('--metric_id', type=int, default=4, help='Metric to be used when ordering frames. 0 - DHW, '
                                                                 '1 - THW, 2 - TTC, 3 or bigger - All metrics '
                                                                 'together ')
    parser.add_argument('--lc', '--lanechanges', action='store_true', help='If set, use only the lane changing tracks.')

    return parser.parse_args()


def generate_cr_scenarios(recording_meta_fn,
                          tracks_meta_fn,
                          tracks_fn,
                          min_time_steps,
                          num_timesteps,
                          num_planning_problems,
                          output_dir,
                          min_ttc_frames_upper,
                          min_ttc_frames_lower):
    """
    Generate CommonRoad xml files with given paths to highD recording, tracks_meta, tracks files
    :param recording_meta_fn: path to *_recordingMeta.csv
    :param tracks_meta_fn: path to *_tracksMeta.csv
    :param tracks_fn: path to *_tracks.csv
    :param min_time_steps: vehicles have to appear more than min_time_steps per .xml to be converted
    :param num_timesteps: maximal number of timesteps per .xml file
    :param num_planning_problems: number of planning problems per .xml file
    :param output_dir: path to store generated .xml files
    :return: None
    """

    def enough_time_steps(vehicle_id, tracks_meta_df, min_time_steps, frame_start, frame_end):
        vehicle_meta = tracks_meta_df[tracks_meta_df.id == vehicle_id]
        if frame_end - int(vehicle_meta.initialFrame) < min_time_steps or \
                int(vehicle_meta.finalFrame) - frame_start < min_time_steps:
            return False
        return True

    # read data frames from three files
    recording_meta_df = pd.read_csv(recording_meta_fn, header=0)
    tracks_meta_df = pd.read_csv(tracks_meta_fn, header=0)
    tracks_df = pd.read_csv(tracks_fn, header=0)

    # generate meta scenario with lanelet network
    meta_scenario_upper, _, _ = get_meta_scenario(recording_meta_df, 1)
    meta_scenario_lower, _, _ = get_meta_scenario(recording_meta_df, 2)

    def create_scenario_with_direction(meta_scenario, min_ttc_frames, direction):
        # for i in range(num_scenarios):
        for track_id, frame_list in min_ttc_frames.items():
            frame_count = len(frame_list)
            for idx, frame in enumerate(frame_list):

                # convert obstacles appearing between [frame_start, frame_end]
                frame_start = frame - num_timesteps
                frame_end = frame + num_timesteps

                ego_final_x_position = tracks_df[(tracks_df.frame == frame_end) & (tracks_df.id == track_id)].x.values
                if len(ego_final_x_position) == 0:
                    continue
                if not 5 < ego_final_x_position[0] < 395:
                    continue

                ego_initial_x_position = tracks_df[
                    (tracks_df.frame == frame_start) & (tracks_df.id == track_id)].x.values
                if len(ego_initial_x_position) == 0:
                    continue
                if not 20 < ego_initial_x_position[0] < 320:
                    continue

                print("\nGenerating scenario {}/{}, vehicle id {}".format(idx + 1, frame_count, track_id))

                # copy meta_scenario with lanelet networks
                scenario = copy.deepcopy(meta_scenario)

                # benchmark id format: COUNTRY_SCENE_CONFIG_PRED
                benchmark_id = "MPP_DEU_{0}-{1}_{2}_T-1".format(
                    location_dict[recording_meta_df.locationId.values[0]], int(recording_meta_df.id),
                    'T%sF%s' % (track_id, frame))
                scenario.benchmark_id = benchmark_id

                # read tracks appear between [frame_start, frame_end]
                scenario_tracks_df = tracks_df[(tracks_df.frame >= frame_start) & (tracks_df.frame <= frame_end)]

                # generate CR obstacles
                for o_idx, vehicle_id in enumerate(scenario_tracks_df.id.unique()):
                    if not tracks_meta_df[tracks_meta_df.id == vehicle_id].drivingDirection.values[0] == direction:
                        continue

                    # if appearing time steps < min_time_steps, skip vehicle
                    if not enough_time_steps(vehicle_id, tracks_meta_df, min_time_steps, frame_start, frame_end):
                        continue

                    do = generate_dynamic_obstacle(scenario, vehicle_id, tracks_meta_df, scenario_tracks_df,
                                                   frame_start,
                                                   track_id)
                    scenario.add_objects(do)

                # generate planning problems
                planning_problem_set = PlanningProblemSet()
                for i in range(num_planning_problems):
                    planning_problem = get_planning_problem(scenario, track_id)
                    planning_problem_set.add_planning_problem(planning_problem)

                print(scenario)
                # write new scenario
                fw = CommonRoadFileWriter(scenario, planning_problem_set, AUTHOR, AFFILIATION, SOURCE, TAGS)
                filename = os.path.join(output_dir, "{}.xml".format(scenario.benchmark_id))
                fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
                print("Scenario file stored in {}".format(filename))

    create_scenario_with_direction(meta_scenario_upper, min_ttc_frames_upper, 1)
    create_scenario_with_direction(meta_scenario_lower, min_ttc_frames_lower, 2)


def filter_tracks_meta(tracks_meta, metric_id=0, lanechanges=False):
    filtered_tracks = tracks_meta[(tracks_meta['numFrames'] >= 100)]
    if metric_id == 0:
        filtered_tracks = filtered_tracks[(filtered_tracks['minDHW'] > 0)]
    elif metric_id == 1:
        filtered_tracks = filtered_tracks[(filtered_tracks['minTHW'] > 0)]
    elif metric_id == 2:
        filtered_tracks = filtered_tracks[(filtered_tracks['minTTC'] > 0)]
    else:
        filtered_tracks = filtered_tracks[(filtered_tracks['minDHW'] > 0)]
        filtered_tracks = filtered_tracks[(filtered_tracks['minTHW'] > 0)]
        filtered_tracks = filtered_tracks[(filtered_tracks['minTTC'] > 0)]

    if lanechanges:
        filtered_tracks = filtered_tracks[(filtered_tracks['numLaneChanges'] > 0)]
    return filtered_tracks[['id', 'finalFrame', 'drivingDirection']].values


def get_frames(tracks, idx, num_timesteps, final_frame, column='dhw'):
    track = tracks[(tracks['id'] == idx) & (tracks[column] > 0)]
    frames = track.sort_values(by=[column]).iloc[:1].frame.values
    valid_frames = [frame for frame in frames if
                    frame - num_timesteps / 2 >= 0 or frame + num_timesteps / 2 <= final_frame]
    return valid_frames


def extract_frames_from_recordings(tracks_meta, tracks, num_timesteps, metric_id=0, lanechanges=False):
    frames_dict_upper = {}
    frames_dict_lower = {}
    for [track_id, final_frame, driving_direction] in filter_tracks_meta(tracks_meta, metric_id, lanechanges):
        if metric_id == 0:
            frames = get_frames(tracks, track_id, num_timesteps, final_frame, 'dhw')
        elif metric_id == 1:
            frames = get_frames(tracks, track_id, num_timesteps, final_frame, 'thw')
        elif metric_id == 2:
            frames = get_frames(tracks, track_id, num_timesteps, final_frame, 'ttc')
        else:
            dhw_frames = get_frames(tracks, track_id, num_timesteps, final_frame, 'dhw')
            thw_frames = get_frames(tracks, track_id, num_timesteps, final_frame, 'thw')
            ttc_frames = get_frames(tracks, track_id, num_timesteps, final_frame, 'ttc')
            frames = dhw_frames + thw_frames + ttc_frames

        if driving_direction == 1:
            frames_dict_upper[track_id] = frames
        else:
            frames_dict_lower[track_id] = frames

    return frames_dict_upper, frames_dict_lower


def generate_cr_scenarios_based_on_metric(recording_meta_fn, tracks_fn, tracks_meta_fn, args, metric_id, lanechanges):
    metric_name = 'min_dhw'
    metric_name = 'min_thw' if metric_id == 1 else metric_name
    metric_name = 'min_ttc' if metric_id == 2 else metric_name

    print('Extracting %s frames from recording {}...'.format(tracks_fn) % metric_name, '\n')
    min_met_frames_upper, min_met_frames_lower = extract_frames_from_recordings(pd.read_csv(tracks_meta_fn),
                                                                                pd.read_csv(tracks_fn),
                                                                                args.num_timesteps,
                                                                                metric_id=metric_id,
                                                                                lanechanges=lanechanges)
    metric_output_path = os.path.join(args.output_dir, metric_name)
    if not os.path.exists(metric_output_path):
        os.makedirs(metric_output_path)

    generate_cr_scenarios(
        recording_meta_fn,
        tracks_meta_fn,
        tracks_fn,
        args.min_time_steps,
        args.num_timesteps,
        args.num_planning_problems,
        metric_output_path,
        min_met_frames_upper,
        min_met_frames_lower
    )


def main():
    start_time = time.time()

    # get arguments
    args = get_args()

    # make output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # generate path to highd data files
    path_tracks = os.path.join(args.highd_dir, "data/*_tracks.csv")
    path_metas = os.path.join(args.highd_dir, "data/*_tracksMeta.csv")
    path_recording = os.path.join(args.highd_dir, "data/*_recordingMeta.csv")

    # get all file names
    listing_tracks = get_file_lists(path_tracks)
    listing_metas = get_file_lists(path_metas)
    listing_recording = get_file_lists(path_recording)

    for index, (recording_meta_fn, tracks_meta_fn, tracks_fn) in enumerate(zip(listing_recording,
                                                                               listing_metas,
                                                                               listing_tracks)):
        # if not index == 11:
        #     continue
        print("=" * 80)
        print("Processing file {}...".format(tracks_fn), '\n')
        print("=" * 80)

        metric_id = args.metric_id
        lanechanges = args.lc

        if 0 <= metric_id <= 2:
            generate_cr_scenarios_based_on_metric(recording_meta_fn, tracks_fn, tracks_meta_fn,
                                                  args, metric_id, lanechanges)
        else:
            generate_cr_scenarios_based_on_metric(recording_meta_fn, tracks_fn, tracks_meta_fn, args, 0, lanechanges)
            generate_cr_scenarios_based_on_metric(recording_meta_fn, tracks_fn, tracks_meta_fn, args, 1, lanechanges)
            generate_cr_scenarios_based_on_metric(recording_meta_fn, tracks_fn, tracks_meta_fn, args, 2, lanechanges)

    print("Elapsed time: {} s".format(time.time() - start_time), "\r")


if __name__ == "__main__":
    main()
