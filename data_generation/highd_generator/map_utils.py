import numpy as np

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import Lanelet

def get_lane_markings(df):
    upper_lane_markings = [-float(x) for x in df.upperLaneMarkings.values[0].split(";")]
    lower_lane_markings = [-float(x) for x in df.lowerLaneMarkings.values[0].split(";")]
    return upper_lane_markings, lower_lane_markings

def get_dt(df):
    return 1./df.frameRate.values[0]

def get_location_id(df):
    return df.locationId.values[0]

def get_file_id(df):
    return df.id.values[0]

def get_speed_limit(df):
    speed_limit = df.speedLimit.values[0]
    if speed_limit < 0:
        return np.inf
    else:
        return speed_limit

def get_meta_scenario(df, direction):
    benchmark_id = "meta_map"
    meta_scenario = Scenario(get_dt(df), benchmark_id)
    upper_lane_markings, lower_lane_markings = get_lane_markings(df)
    speed_limit = get_speed_limit(df)

    if direction == 1:
        for i in range(len(upper_lane_markings)-1):

            # get two lines of current lane
            next_lane_y = upper_lane_markings[i+1]
            lane_y = upper_lane_markings[i]

            # get vertices of three lines
            right_vertices = np.array([[400., lane_y], [0., lane_y]])
            left_vertices = np.array([[400., next_lane_y], [0., next_lane_y]])
            center_vertices = (left_vertices + right_vertices) / 2.

            # assign lane ids and adjacent ids
            lanelet_id = i + 1
            adjacent_left = lanelet_id + 1
            adjacent_right = lanelet_id - 1
            adjacent_left_same_direction = True
            adjacent_right_same_direction = True
            if i == 0:
                adjacent_right = None
            elif i == len(lower_lane_markings)-1:
                adjacent_left = None

            # add lanelet to scenario
            meta_scenario.add_objects(
                Lanelet(
                    lanelet_id=lanelet_id,
                    left_vertices=left_vertices,
                    right_vertices=right_vertices,
                    center_vertices=center_vertices,
                    adjacent_left=adjacent_left,
                    adjacent_left_same_direction=adjacent_left_same_direction,
                    adjacent_right=adjacent_right,
                    adjacent_right_same_direction=adjacent_right_same_direction,
                    speed_limit=speed_limit
                )
            )

    if direction == 2:
        for i in range(len(lower_lane_markings)-1):

            # get two lines of current lane
            next_lane_y = lower_lane_markings[i+1]
            lane_y = lower_lane_markings[i]

            # get vertices of three lines
            left_vertices = np.array([[0., lane_y], [400., lane_y]])
            right_vertices = np.array([[0., next_lane_y], [400., next_lane_y]])
            center_vertices = (left_vertices + right_vertices) / 2.

            # assign lane ids and adjacent ids
            lanelet_id = i + 1 + len(upper_lane_markings) -1
            adjacent_left = lanelet_id - 1
            adjacent_right = lanelet_id + 1
            adjacent_left_same_direction = True
            adjacent_right_same_direction = True
            if i == 0:
                adjacent_left = None
            elif i == len(lower_lane_markings)-1:
                adjacent_right = None

            # add lanelet to scenario
            meta_scenario.add_objects(
                Lanelet(
                    lanelet_id=lanelet_id,
                    left_vertices=left_vertices,
                    right_vertices=right_vertices,
                    center_vertices=center_vertices,
                    adjacent_left=adjacent_left,
                    adjacent_left_same_direction=adjacent_left_same_direction,
                    adjacent_right=adjacent_right,
                    adjacent_right_same_direction=adjacent_right_same_direction,
                    speed_limit=speed_limit
                )
            )

    return meta_scenario, upper_lane_markings, lower_lane_markings