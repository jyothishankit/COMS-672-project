from constants import *
from hyperparameters import *
import _pickle as pickle
import numpy as np
from vehicle import *
import pandas as pd

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

"""
self.current_time += TIMESTEP
        self.minofday += int(TIMESTEP / 60.0)
        if self.minofday >= 1440:
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7
"""
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