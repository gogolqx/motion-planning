import os
import random
import time
from typing import Tuple, List, Union, Dict, Callable

import commonroad
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.prediction.prediction import SetBasedPrediction, TrajectoryPrediction
from commonroad.scenario.trajectory import State
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.visualization.planning import draw_func_dict as planning_draw_func_dict, draw_goal_region
from commonroad.visualization.scenario import draw_func_dict as scenario_draw_func_dict, draw_rectangle
from matplotlib.animation import FuncAnimation


def visualize_scenario(file_name, scenario_files_path, single_frame=False):
    file_path = scenario_files_path + '/' + file_name
    save_path = scenario_files_path + '/' + file_name.replace('.xml', '.gif')

    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    print('Visualizing scenario %s.' % scenario.benchmark_id)

    plot_limits = get_plot_limits(scenario, planning_problem_set)
    figsize = get_figsize(plot_limits)
    frame_count = get_frame_count(scenario)
    interval = 1000 * scenario.dt  # delay between frames in milliseconds, 1 second * dt to get actual time in ms
    frame_count += int(1 / (scenario.dt * 2))  # add short padding to create a short break before the loop (half sec)

    fig = plt.figure(figsize=figsize)
    ln = plt.plot([], [], animated=True)

    handles = {}

    # Separate the ego vehicle from dynamic obstacles and draw it separately
    ego_vehicle = scenario.obstacle_by_id(99)  # ego id is always 99
    scenario.remove_obstacle(ego_vehicle)

    # Initialize the animator
    draw_object(scenario.lanelet_network, plot_limits=plot_limits)
    draw_object(scenario.static_obstacles, plot_limits=plot_limits)
    draw_object(planning_problem_set, plot_limits=plot_limits,
                draw_func=planning_problem_set_funcs())

    if single_frame:
        draw_object(scenario.dynamic_obstacles, plot_limits=plot_limits, handles=handles, draw_params=scenario_params(
            time_begin=40, time_end=41))
        draw_object(ego_vehicle, plot_limits=plot_limits, handles=handles, draw_params=ego_params(
            time_begin=40, time_end=41))
        plt.savefig(scenario_files_path + '/' + file_name.replace('.xml', '.png'))
        return

    def init():
        draw_object(scenario.dynamic_obstacles, plot_limits=plot_limits, handles=handles,
                    draw_params=scenario_params(time_begin=0, time_end=1))
        return ln

    def animate(frame):
        # Update function for the animator
        for handles_i in handles.values():
            for handle in handles_i:
                if handle is not None:
                    handle.remove()
        handles.clear()

        draw_object(scenario.dynamic_obstacles, plot_limits=plot_limits, handles=handles, draw_params=scenario_params(
            time_begin=frame, time_end=min(frame_count, frame + 1)))
        draw_object(ego_vehicle, plot_limits=plot_limits, handles=handles, draw_params=ego_params(
            time_begin=frame, time_end=min(frame_count, frame + 1)))
        return ln

    anim = FuncAnimation(fig, animate, frames=frame_count,
                         init_func=init, blit=True, interval=interval)
    anim.save(save_path, dpi=100, writer='imagemagick')
    plt.close(plt.gcf())


def get_plot_limits(scenario, planning_problem_set, padding=10):
    #  Get lanelet limits
    lanelet_limits = get_plot_limits_by_lanelet_network(scenario)
    return add_padding(lanelet_limits)


def add_padding(limits, padding=20):
    return [int(limits[0]) - padding, int(limits[1] + padding), int(limits[2]) - padding, int(limits[3]) + padding]


def get_plot_limits_by_lanelet_network(scenario):
    lanelets = scenario.lanelet_network.lanelets

    left_x = set(left_vertice[0] for lanelet in lanelets for left_vertice in lanelet.left_vertices)
    left_y = set(left_vertice[1] for lanelet in lanelets for left_vertice in lanelet.left_vertices)
    right_x = set(right_vertice[0] for lanelet in lanelets for right_vertice in lanelet.right_vertices)
    right_y = set(right_vertice[1] for lanelet in lanelets for right_vertice in lanelet.right_vertices)

    x_points, y_points = set.union(left_x, right_x), set.union(left_y, right_y)

    min_x, max_x, min_y, max_y = min(x_points), max(x_points), min(y_points), max(y_points)

    return [min_x, max_x, min_y, max_y]


def get_figsize(plot_limits, max_height=8, max_width=8):
    x_size = plot_limits[1] - plot_limits[0]
    y_size = plot_limits[3] - plot_limits[2]

    if x_size > y_size:
        figsize_x = max_width
        figsize_y = max_height * y_size / x_size
        return figsize_x + 3, figsize_y + 1

    if y_size > x_size:
        figsize_x = max_width * x_size / y_size
        figsize_y = max_height
        return figsize_x + 1, figsize_y + 3

    return max_height, max_width


def get_frame_count(scenario):
    """
    Calculates frame count for a scenario. This is the number of timesteps of the longest moving obstacle.
    """
    frame_count = 1
    obstacles = scenario.dynamic_obstacles
    for o in obstacles:
        if type(o.prediction) == SetBasedPrediction:
            frame_count = max(frame_count, len(o.prediction.occupancy_set))
        elif type(o.prediction) == TrajectoryPrediction:
            frame_count = max(frame_count, len(o.prediction.trajectory.state_list))
    return frame_count


def planning_problem_set_funcs():
    custom_func_dict = scenario_draw_func_dict.copy()
    custom_func_dict.update(planning_draw_func_dict)
    custom_func_dict[PlanningProblemSet] = draw_planning_problem_set
    return custom_func_dict


def scenario_params(time_begin=0, time_end=1):
    return {
        'time_begin': time_begin,
        'time_end': time_end,
        'antialiased': True,
    }


def ego_params(time_begin=0, time_end=1):
    return {
        'time_begin': time_begin,
        'time_end': time_end,
        'antialiased': True,
        'dynamic_obstacle': {
            'shape': {
                'rectangle': {
                    'facecolor': '#bf1d00',
                    'edgecolor': '#000000',
                },
            },
        },
    }


'''
Modified PlanningProblemSet Drawers

Except the draw_initial_state function, all others are have the same functionality as the functions with
the same name on commonroad.visualization.planning file. 

Even if we set State: draw_initial_state on the functions dictionary it is not called after the PlanningProblemSet
one is called. That's why we need to modify all of them starting from draw_planning_problem_set
'''


def draw_initial_state(obj: State, plot_limits: List[Union[int, float]],
                       ax: mpl.axes.Axes, draw_params: dict, draw_func: Dict[type, Callable],
                       handles: Dict[int, List[mpl.patches.Patch]], call_stack: Tuple[str, ...]) -> None:
    circle_radius = 1.4
    arrow_length = 6
    arrow_width = 0.2
    color = 'g'
    position = obj.position
    orientation = obj.orientation
    length_x = np.math.cos(orientation) * arrow_length
    length_y = np.math.sin(orientation) * arrow_length
    offset_x = position[0] + np.math.cos(orientation) * circle_radius
    offset_y = position[1] + np.math.sin(orientation) * circle_radius

    initial_state_circle = plt.Circle(position, circle_radius, color=color, zorder=30, alpha=.5)
    ax.add_artist(initial_state_circle)
    ax.arrow(offset_x, offset_y, length_x, length_y, color=color, width=arrow_width, zorder=30, alpha=.5)


def draw_planning_problem(obj: PlanningProblem, plot_limits: List[Union[int, float]],
                          ax: mpl.axes.Axes, draw_params: dict, draw_func: Dict[type, Callable],
                          handles: Dict[int, List[mpl.patches.Patch]], call_stack: Tuple[str, ...]) -> None:
    call_stack = tuple(list(call_stack) + ['planning_problem'])
    if not 'initial_state' in draw_params:
        draw_params['initial_state'] = {}
    draw_params['initial_state']['label'] = 'initial position'
    draw_initial_state(obj.initial_state, plot_limits, ax, draw_params, draw_func, handles, call_stack)
    draw_goal_region(obj.goal, plot_limits, ax, draw_params, draw_func, handles, call_stack)


def draw_planning_problem_set(obj: PlanningProblemSet, plot_limits: List[Union[int, float]],
                              ax: mpl.axes.Axes, draw_params: dict, draw_func: Dict[type, Callable],
                              handles: Dict[int, List[mpl.patches.Patch]], call_stack: Tuple[str, ...]) -> None:
    call_stack = tuple(list(call_stack) + ['planning_problem_set'])
    try:
        draw_ids = commonroad.visualization.draw_dispatch_cr._retrieve_value(
            draw_params, call_stack,
            tuple(['draw_ids']))
    except KeyError:
        print("Cannot find stylesheet for planning_problem. Called through:")
        print(call_stack)

    for id, problem in obj.planning_problem_dict.items():
        if draw_ids is 'all' or id in draw_ids:
            draw_planning_problem(problem, plot_limits, ax, draw_params, draw_func, handles, call_stack)


"""
Look for files and visualize
"""
print('Visualizing scenarios...')

print('Using Agg backend for faster visualization.')
mpl.use('Agg')

print('Using "classic" for plot style.')
plt.style.use('classic')

scenario_files_path = os.getcwd() + '/scenarios'
files = sorted(os.listdir(scenario_files_path))
xml_files = [file for file in files if file.endswith('.xml')]
selected_files = random.sample(xml_files, k=100)

start = time.time()
file_counter = 0
for file in selected_files:
    visualize_scenario(file, scenario_files_path, single_frame=False)
    file_counter += 1
end = time.time()

print('Done! It took %s seconds to visualize %s scenario files.' % (end - start, file_counter))
