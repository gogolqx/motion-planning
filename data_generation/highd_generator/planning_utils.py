import random
from commonroad.scenario.trajectory import State
from commonroad.common.util import Interval, AngleInterval
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.goal import GoalRegion


def get_planning_problem(scenario,
                         ego_id,
                         orientation_half_range=0.2, 
                         velocity_half_range=10., 
                         time_step_half_range=50,
):    
    dynamic_obstacle = scenario.obstacle_by_id(99)  # ego id is always 99
    dynamic_obstacle_final_state = dynamic_obstacle.prediction.trajectory.final_state
    # scenario.remove_obstacle(dynamic_obstacle)
    
    # define orientation, velocity and time step intervals as goal region
    orientation_interval = AngleInterval(
        dynamic_obstacle_final_state.orientation - orientation_half_range,
        dynamic_obstacle_final_state.orientation + orientation_half_range
    )
    velocity_interval = Interval(
        dynamic_obstacle_final_state.velocity - velocity_half_range,
        dynamic_obstacle_final_state.velocity + velocity_half_range
    )
    time_step_interval = Interval(
        dynamic_obstacle_final_state.time_step - time_step_half_range,
        dynamic_obstacle_final_state.time_step + time_step_half_range
    )
    goal_region = GoalRegion([State(
        orientation=orientation_interval, 
        velocity=velocity_interval, 
        time_step=time_step_interval
    )])
    
    dynamic_obstacle.initial_state.yaw_rate = 0.
    dynamic_obstacle.initial_state.slip_angle = 0.
    
    return PlanningProblem(dynamic_obstacle.obstacle_id, dynamic_obstacle.initial_state, goal_region)
