import numpy as np
import pandas as pd
from pathgenerator import PathGenerator
import geohelper as gh
from hyperparameters import *
from vehicle import *
from utils import *

class FleetSimulator(object):
    def __init__(self, G, eta_model, cycle, max_action_time=20):
        self.router = PathGenerator(G)
        self.max_action_time = max_action_time
        self.eta_model = eta_model
        self.cycle = cycle

    def reset(self, num_vehicles, dataset, dayofweek, minofday):
        init_locations = get_initial_locations(dataset, num_vehicles)
        self.requests = dataset
        self.vehicles = get_initial_location_vehicles(init_locations, num_vehicles)
        self.current_time = 0
        self.minofday = minofday
        self.dayofweek = dayofweek


    def get_requests(self, num_steps, offset=0):
        requests = get_requests_with_offset(self.requests, self.current_time, num_steps, offset)
        return requests

    def calculate_num_steps(self):
        return get_number_of_steps(self.cycle)

    def process_actions(self, actions):
        if actions:
            self.dispatch(actions)

    def get_and_process_requests(self, num_steps):
        requests = self.get_requests(num_steps)
        current_time = self.current_time
        current_requests = get_requests_without_offset(requests, current_time)
        return current_requests

    def run_step_iterations(self, num_steps, current_requests):
        wait = 0
        reject = 0
        gas = 0
        request_hop_zero = 0
        for _ in range(num_steps):
            for vehicle in self.vehicles:
                gas += vehicle.transition()
            vehicle_locations = self.get_vehicles_location()
            assignments = self.match(vehicle_locations, current_requests)
            current_wait, num_vehicles, num_passengers, request_hop_zero_current = self.assign(assignments)
            reject += len(current_requests) - num_passengers
            wait += current_wait
            request_hop_zero += request_hop_zero_current
            self.update_time()
        return wait, reject, gas, request_hop_zero

    def step(self, actions=None):
        num_steps = self.calculate_num_steps()
        self.process_actions(actions)
        current_requests = self.get_and_process_requests(num_steps)
        wait, reject, gas, request_hop_zero = self.run_step_iterations(num_steps, current_requests)
        vehicles = self.get_vehicles_dataframe()
        return vehicles, current_requests, wait, reject, gas, request_hop_zero
    
    def get_vehicles_dataframe(self):
        vehicles = [vehicle.get_state() for vehicle in self.vehicles]
        vehicles = get_vehicle_dataframe_with_columns(vehicles)
        return vehicles

    def update_time(self):
        self.current_time = get_updated_current_time(self.current_time)
        self.minofday, self.dayofweek = get_updated_minofday_dayofweek(self.minofday, self.dayofweek)

    def get_vehicles_score(self):
        vehicles = get_vehicle_reward_service_time_idle_time(vehicles)
        return vehicles

    def match(self, resources, tasks):
        available_resources = find_available_resources(resources)
        unique_tasks = count_unique_tasks(tasks)
        task_indices = find_task_indices(tasks, unique_tasks)
        assignments = assign_resources_to_tasks(available_resources, unique_tasks, task_indices)
        return assignments

    def assign(self, assignments):
        num_vehicles_assigned = len(assignments)
        hop_zero_requests = 0 
        total_passengers = 0
        total_wait_time = 0
        for request_idx, vehicle_idx in assignments:
            vehicle = self.vehicles[vehicle_idx]
            request = self.requests.loc[request_idx]
            num_passengers_in_vehicle = len(request) 
            hop_zero_requests += count_hopper_flag_zero(request)
            total_passengers += num_passengers_in_vehicle
            request_sorted = sort_request_by_trip_time(request)
            first_request = request_sorted.iloc[0]
            last_request = request_sorted.iloc[-1]
            pickup_location = (first_request.plat, first_request.plon)
            dropoff_location = (last_request.dlat, last_request.dlon)
            vehicle_location = vehicle.location
            distance_to_pickup = calculate_distance(vehicle_location, pickup_location)
            wait_time = calculate_wait_time(distance_to_pickup)
            effective_distance = sum(request.trip_distance)
            actual_distance, trip_time = calculate_actual_trip_time(request_sorted)
            vehicle.start_service(dropoff_location, wait_time, trip_time, actual_distance, effective_distance, num_passengers_in_vehicle)
            total_wait_time += wait_time

        return total_wait_time, num_vehicles_assigned, total_passengers, hop_zero_requests

    def dispatch(self, actions):
        vehicle_locations, target_locations = prepare_vehicle_and_target_locations(self, actions)
        distances = calculate_distances(self, vehicle_locations, target_locations)
        feature_matrix = prepare_feature_matrix(self, vehicle_locations, target_locations, distances)
        trip_times = self.eta_model.predict(feature_matrix)
        execute_actions(self, actions, trip_times)


    def get_vehicles_location(self):
        vehicles = [vehicle.get_location() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'lat', 'lon', 'available'])
        return vehicles

