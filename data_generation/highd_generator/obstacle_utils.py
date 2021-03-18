import scipy.signal
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

obstacle_class_dict = {
	'Truck': ObstacleType.TRUCK,
	'Car': ObstacleType.CAR
}

def savgol_filter(signal, polyorder=7):
    window_size = len(signal) // 4
    if window_size % 2 == 0:
        window_size -= 1
    if window_size <= polyorder:
        return None
    else:
        return scipy.signal.savgol_filter(signal, window_size, polyorder=polyorder)

def filter_signal(signal, polyorder=None):

	if polyorder is None: 
	     # no filter
	    return np.array(signal)
	else:
	    # apply savgol filter with polyorder
	    return savgol_filter(signal, polyorder=polyorder)	       

def get_velocity(df):
    return np.sqrt(df.xVelocity**2 + df.yVelocity**2)

def get_orientation(df):
    return np.arctan2(-df.yVelocity, df.xVelocity)

def get_acceleration(df):
    return np.sqrt(df.xAcceleration**2 + df.yAcceleration**2)

def generate_dynamic_obstacle(scenario, vehicle_id, tracks_meta_df, tracks_df, frame_start, ego_id):
    
    vehicle_meta = tracks_meta_df[tracks_meta_df.id == vehicle_id]
    vehicle_tracks = tracks_df[tracks_df.id == vehicle_id]

    length = vehicle_meta.width.values[0]
    width = vehicle_meta.height.values[0]

    initial_time_step = int(vehicle_tracks.frame.values[0]) - int(frame_start)
    dynamic_obstacle_id = int(vehicle_id)*10 if not vehicle_id == ego_id else 99 # set ego id as 99, and append 0 for others
    dynamic_obstacle_type = obstacle_class_dict[vehicle_meta['class'].values[0]]
    dynamic_obstacle_shape = Rectangle(width=width, length=length)

    polyorder = None
    xs = filter_signal(vehicle_tracks.x, polyorder=None) # checked x signals, no need to filter
    ys = filter_signal(-vehicle_tracks.y - 1.2, polyorder=polyorder)
    velocities = filter_signal(get_velocity(vehicle_tracks), polyorder=polyorder)
    orientations = filter_signal(get_orientation(vehicle_tracks), polyorder=polyorder)
    accelerations = filter_signal(get_acceleration(vehicle_tracks), polyorder=polyorder)

    state_list = []
    for i, (x, y, v, theta, a) in enumerate(
        zip(xs, ys, velocities, orientations, accelerations)
    ):
        state_list.append(State(
            position=np.array([x, y]),
            velocity=v,
            orientation=theta,
            time_step=initial_time_step+i
        ))
    dynamic_obstacle_initial_state = state_list[0]
    dynamic_obstacle_trajectory = Trajectory(initial_time_step+1, state_list[1:])
    dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)
    return DynamicObstacle(
        dynamic_obstacle_id,
        dynamic_obstacle_type,
        dynamic_obstacle_shape,
        dynamic_obstacle_initial_state,
        dynamic_obstacle_prediction
    )