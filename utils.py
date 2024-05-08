from constants import *
from hyperparameters import *
import _pickle as pickle
import numpy as np
from vehicle import *
import pandas as pd
import geohelper as gh

def load_graph():
    with open(GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)
    return G

def load_eta_model():
    with open(ETA_MODEL_PATH, 'rb') as f:
        eta_model = pickle.load(f)
    return eta_model

def get_initial_locations(dataset, num_vehicles):
    return dataset[['plat', 'plon']].values[np.arange(num_vehicles) % len(dataset)]

def get_initial_location_vehicles(init_locations, num_vehicles):
    return [Vehicle(i, init_locations[i]) for i in range(num_vehicles)]

def get_requests_with_offset(requests, current_time, num_steps, offset):
    return requests[(requests.second >= current_time + offset * TIMESTEP)
                                 &(requests.second < current_time + TIMESTEP * (num_steps + offset))]

def get_requests_without_offset(requests, current_time):
    return requests[(requests.second >= current_time)
                                 &(requests.second < current_time + TIMESTEP)]

def get_number_of_steps(cycle):
    return int(cycle * 60.0 / TIMESTEP)

def get_vehicle_dataframe_with_columns(vehicles):
    return pd.DataFrame(vehicles, columns=['id', 'available', 'geohash', 'dest_geohash',
                                                   'eta', 'status', 'reward', 'lat', 'lon', 'idle','eff_dist','act_dist'])

def get_updated_current_time(current_time):
    return current_time+TIMESTEP

def get_updated_minofday_dayofweek(minofday, dayofweek):
    minofday += int(TIMESTEP / 60.0)
    if minofday >= 1440:
        minofday -= 1440
        dayofweek = (dayofweek + 1) % 7
    return minofday, dayofweek

def get_vehicle_reward_service_time_idle_time(vehicles):
    vehicles_dataframe_input =[vehicle.get_score() for vehicle in vehicles]
    vehicles_dataframe = pd.DataFrame(vehicles_dataframe_input, columns=['id', 'reward', 'service_time', 'idle_time'])
    return vehicles_dataframe

def find_available_resources(resources):
        available_resources = resources[resources.available == 1]
        return available_resources

def count_unique_tasks(tasks):
    unique_tasks = tasks.groupby(['plat','plon']).count()
    unique_tasks = unique_tasks.reset_index(level=['plat','plon'])
    unique_tasks = unique_tasks[['plat','plon']]
    return unique_tasks

def find_task_indices(tasks, unique_tasks):
    task_indices = unique_tasks.apply(lambda row: tasks[(tasks['plat']==row['plat']) & 
                                                        (tasks['plon']==row['plon'])].index.values, 
                                    axis=1)
    return task_indices

def assign_resources_to_tasks(available_resources, unique_tasks, task_indices):
    distances = gh.distance_in_meters(available_resources.lat.values,
                                    available_resources.lon.values,
                                    unique_tasks.plat.values[:, None],
                                    unique_tasks.plon.values[:, None])
    num_tasks = min(len(unique_tasks), len(available_resources))
    assigned_resources = np.zeros(num_tasks, dtype=int)
    for i in range(num_tasks):
        closest_resource_index = distances[i].argmin()
        if distances[i, closest_resource_index] > REJECT_DISTANCE:
            assigned_resources[i] = -1
        else:
            assigned_resources[i] = closest_resource_index
            distances[:, closest_resource_index] = float('inf')
    task_indices_to_use = unique_tasks.index[:num_tasks][assigned_resources >= 0]
    assignments = list(zip(unique_tasks.loc[task_indices_to_use]['index'], 
                        available_resources['id'].iloc[assigned_resources[assigned_resources >= 0]]))
    return assignments


def count_hopper_flag_zero(request):
        return sum(request['hop_flag'] == 0)

def sort_request_by_trip_time(request):
    return request.sort_values(by=['trip_time'])

def calculate_distance(loc1, loc2):
    return gh.distance_in_meters(loc1[0], loc1[1], loc2[0], loc2[1])

def calculate_wait_time(distance):
    return (distance * 2 / 1.414) / (ASSIGNMENT_SPEED * 1000 / 60)

def calculate_actual_trip_time(request_sorted):
    request_copy = request_sorted.copy()
    request_sorted.loc[:, 'index_val'] = range(len(request_sorted))
    request_copy.loc[:, 'index_val'] = request_sorted['index_val'].values - 1
    request_copy = request_copy[['dlat','dlon','index_val']]
    request_join = pd.merge(request_sorted, request_copy, how='left', on='index_val')
    request_join_no_na = request_join.dropna()
    if len(request_join_no_na) > 0:
        request_join_no_na['dist_bw_dest'] = gh.distance_in_meters(request_join_no_na.dlat_x,request_join_no_na.dlon_x,
                                                                request_join_no_na.dlat_y,request_join_no_na.dlon_y)
        actual_distance = request_sorted.iloc[0].trip_distance + (sum(request_join_no_na.dist_bw_dest)) / 1000
        trip_time = (actual_distance / ASSIGNMENT_SPEED) * 60
    else:
        actual_distance = request_sorted.iloc[0].trip_distance
        trip_time = request_sorted.iloc[0].trip_time
    return actual_distance, trip_time


def prepare_vehicle_and_target_locations(self, actions):
        vehicle_locations = [self.vehicles[vid].location for vid, _ in actions]
        target_locations = [target for _, target in actions]
        return vehicle_locations, target_locations

def calculate_distances(self, vehicle_locations, target_locations):
    distances = []
    for vloc, tloc in zip(vehicle_locations, target_locations):
        try:
            path, distance, _, _ = self.router.map_matching_shortest_path(vloc, tloc)
            distances.append(distance)
        except:
            start_lat, start_lon = vloc
            dest_lat, dest_lon = tloc
            d = gh.distance_in_meters(start_lat, start_lon, dest_lat, dest_lon)
            distances.append(d)
    return distances

def prepare_feature_matrix(self, vehicle_locations, target_locations, distances):
    N = len(vehicle_locations)
    X = np.zeros((N, 7))
    X[:, 0] = self.dayofweek
    X[:, 1] = self.minofday / 60.0
    X[:, 2:4] = vehicle_locations
    X[:, 4:6] = target_locations
    X[:, 6] = distances
    return X

def execute_actions(self, actions, trip_times):
    for i, (vid, _) in enumerate(actions):
        if trip_times[i] > MIN_TRIPTIME:
            eta = min(trip_times[i], self.max_action_time)
            self.vehicles[vid].route([], eta)

def get_vehicle_locations_from_dataframe(vehicles):
    vehicles_dataframe_input = [vehicle.get_location() for vehicle in vehicles]
    vehicles_location_dataframe = pd.DataFrame(vehicles_dataframe_input, columns=['id', 'lat', 'lon', 'available'])
    return vehicles_location_dataframe