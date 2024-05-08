from constants import *
from hyperparameters import *
import _pickle as pickle
import numpy as np
from vehicle import *
import pandas as pd
import geohelper as geo_h

def load_graph():
    with open(GRAPH_PATH, 'rb') as file:
        G = pickle.load(file)
    return G

def load_eta_model():
    with open(ETA_MODEL_PATH, 'rb') as file:
        eta_model = pickle.load(file)
    return eta_model

def get_initial_locations(data, number_of_vehicles):
    return data[['plat', 'plon']].values[np.arange(number_of_vehicles) % len(data)]

def get_initial_location_vehicles(initial_locations, number_of_vehicles):
    return [Vehicle(i, initial_locations[i]) for i in range(number_of_vehicles)]

def get_requests_with_offset(reqs, cur_time, num_steps, off):
    return reqs[(reqs.second >= cur_time + off * TIMESTEP)
                & (reqs.second < cur_time + TIMESTEP * (num_steps + off))]

def get_requests_without_offset(reqs, cur_time):
    return reqs[(reqs.second >= cur_time)
                & (reqs.second < cur_time + TIMESTEP)]

def get_number_of_steps(cycle):
    return int(cycle * 60.0 / TIMESTEP)

def get_vehicle_dataframe_with_columns(vehicles_list_csv):
    column_list = ['id', 'available', 'geohash', 'dest_geohash', 'eta', 'status', 'reward', 'lat', 'lon', 'idle','eff_dist','act_dist']
    return pd.DataFrame(vehicles_list_csv, columns=column_list)

def get_updated_current_time(current_time):
    return current_time+TIMESTEP

def get_updated_minofday_dayofweek(minute_of_day, day_of_week):
    minute_of_day += int(TIMESTEP / 60.0)
    if minute_of_day >= 1440:
        minute_of_day -= 1440
        day_of_week = (day_of_week + 1) % 7
    return minute_of_day, day_of_week

def get_vehicle_reward_service_time_idle_time(vehicles):
    vehicles_dataframe_input =[vehicle.get_score() for vehicle in vehicles]
    column_list = ['id', 'reward', 'service_time', 'idle_time']
    vehicles_dataframe = pd.DataFrame(vehicles_dataframe_input, columns=column_list)
    return vehicles_dataframe

def find_available_resources(resources):
        available_resources = resources[resources.available == 1]
        return available_resources

def count_unique_tasks(task_list):
    unique_tasks = task_list.groupby(['plat','plon']).count()
    unique_tasks = unique_tasks.reset_index(level=['plat','plon'])
    unique_tasks = unique_tasks[['plat','plon']]
    return unique_tasks

def find_task_indices(task_list, unique_task_list):
    task_indices = unique_task_list.apply(lambda row: task_list[(task_list['plat']==row['plat']) & 
                                                                (task_list['plon']==row['plon'])].index.values, 
                                          axis=1)
    return task_indices


def assign_resources_to_tasks(available_resources, unique_tasks, task_indices):
    distances = geo_h.distance_in_meters(available_resources.lat.values,
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
    return geo_h.distance_in_meters(loc1[0], loc1[1], loc2[0], loc2[1])

def calculate_wait_time(distance):
    return (distance * 2 / 1.414) / (ASSIGNMENT_SPEED * 1000 / 60)

def calculate_actual_trip_time(sorted_request):
    request_sorted_copy= sorted_request.copy()
    sorted_request.loc[:, 'index_value'] = range(len(sorted_request))
    request_sorted_copy.loc[:, 'index_value'] = sorted_request['index_value'].values - 1
    request_sorted_copy= request_sorted_copy[['dlat','dlon','index_value']]
    sorted_join_requests = pd.merge(sorted_request, request_sorted_copy, how='left', on='index_value')
    sorted_join_requests_non_empty = sorted_join_requests.dropna()
    if len(sorted_join_requests_non_empty) > 0:
        sorted_join_requests_non_empty['dist_bw_dest'] = geo_h.distance_in_meters(sorted_join_requests_non_empty.dlat_x,sorted_join_requests_non_empty.dlon_x,
                                                                sorted_join_requests_non_empty.dlat_y,sorted_join_requests_non_empty.dlon_y)
        km_distance_actual = sorted_request.iloc[0].trip_distance + (sum(sorted_join_requests_non_empty.distance_between_destinations)) / 1000
        trip_actual_time = (km_distance_actual / ASSIGNMENT_SPEED) * 60
    else:
        actual_distance = sorted_request.iloc[0].trip_distance
        trip_actual_time = sorted_request.iloc[0].trip_time
    return actual_distance, trip_actual_time


def prepare_vehicle_and_target_locations(self, actions):
        vehicle_locations = [self.vehicles[vid].location for vid, _ in actions]
        target_locations = [target for _, target in actions]
        return vehicle_locations, target_locations

def calculate_distances(self, vehicle_locations, target_locations):
    distance_list = []
    for vehicle_location, target_location in zip(vehicle_locations, target_locations):
        try:
            path, distance, _, _ = self.router.map_matching_shortest_path(vehicle_location, target_location)
            distance_list.append(distance)
        except:
            start_latitude, start_longitude = vehicle_location
            dest_latitude, dest_longitude = target_location
            d = geo_h.distance_in_meters(start_latitude, start_longitude, dest_latitude, dest_longitude)
            distance_list.append(d)
    return distance_list

def prepare_feature_matrix(self, vehicle_locs, target_locs, dists):
    num_vehicles = len(vehicle_locs)
    feature_matrix = np.zeros((num_vehicles, 7))
    feature_matrix[:, 0] = self.dayofweek
    feature_matrix[:, 1] = get_hour_of_day(self.minofday)
    feature_matrix[:, 2:4] = vehicle_locs
    feature_matrix[:, 4:6] = target_locs
    feature_matrix[:, 6] = dists
    return feature_matrix

def get_hour_of_day(minute_of_day):
    return minute_of_day/60.0

def execute_actions(self, action_list, time__trip_list):
    for i, (vehicle_id, _) in enumerate(action_list):
        if time__trip_list[i] > MIN_TRIPTIME:
            estimated_time = min(time__trip_list[i], self.max_action_time)
            self.vehicles[vehicle_id].route([], estimated_time)


def get_vehicle_locations_from_dataframe(vehicles):
    vehicles_dataframe_input = [vehicle.get_location() for vehicle in vehicles]
    column_list = ['id', 'lat', 'lon', 'available']
    vehicles_location_dataframe = pd.DataFrame(vehicles_dataframe_input, columns=column_list)
    return vehicles_location_dataframe