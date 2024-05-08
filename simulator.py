from pathgenerator import PathGenerator
from hyperparameters import *
from vehicle import * 
from utils import *

class FleetSimulator(object):
    def __init__(self, graph, model_eta, duration_cycle, max_action_time=20):
        self.router = PathGenerator(graph)
        self.max_action_time = max_action_time
        self.eta_model = model_eta
        self.cycle = duration_cycle

    def reset(self, vehicle_count, data_set, day_of_week, minute_of_day):
        initial_locations = get_initial_locations(data_set, vehicle_count)
        self.requests = data_set
        self.vehicles = get_initial_location_vehicles(initial_locations, vehicle_count)
        self.current_time = 0
        self.minofday = minute_of_day
        self.dayofweek = day_of_week


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

    def run_step_iterations(self, number_of_steps, current_service_requests):
        total_wait_time = 0
        total_rejected_requests = 0
        total_gas_consumption = 0
        total_requests_with_zero_hops = 0
        for _ in range(number_of_steps):
            for vehicle in self.vehicles:
                total_gas_consumption += vehicle.transition()
            vehicle_locations = self.get_vehicles_location()
            assignments = self.match(vehicle_locations, current_service_requests)
            current_wait_time, number_of_vehicles, number_of_passengers, requests_with_zero_hops = self.assign(assignments)
            total_rejected_requests += len(current_service_requests) - number_of_passengers
            total_wait_time += current_wait_time
            total_requests_with_zero_hops += requests_with_zero_hops
            self.update_time()
        return total_wait_time, total_rejected_requests, total_gas_consumption, total_requests_with_zero_hops

    def step(self, actions=None):
        total_steps = self.calculate_num_steps()
        self.process_actions(actions)
        service_requests = self.get_and_process_requests(total_steps)
        total_waiting_time, total_rejected_requests, total_gas_consumption, requests_with_zero_hops = self.run_step_iterations(total_steps, service_requests)
        vehicle_data = self.get_vehicles_dataframe()
        return vehicle_data, service_requests, total_waiting_time, total_rejected_requests, total_gas_consumption, requests_with_zero_hops
    
    def get_vehicles_dataframe(self):
        vehicle_states = [vehicle.get_state() for vehicle in self.vehicles]
        vehicles_dataframe = get_vehicle_dataframe_with_columns(vehicle_states)
        return vehicles_dataframe

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

    def assign(self, vehicle_assignments):
        total_assigned_vehicles = len(vehicle_assignments)
        zero_hop_requests_count = 0 
        total_passenger_count = 0
        total_waiting_time = 0
        for request_index, vehicle_index in vehicle_assignments:
            current_vehicle = self.vehicles[vehicle_index]
            current_request = self.requests.loc[request_index]
            passenger_count_in_vehicle = len(current_request) 
            zero_hop_requests_count += count_hopper_flag_zero(current_request)
            total_passenger_count += passenger_count_in_vehicle
            sorted_request = sort_request_by_trip_time(current_request)
            first_request = sorted_request.iloc[0]
            last_request = sorted_request.iloc[-1]
            pickup_location = (first_request.plat, first_request.plon)
            dropoff_location = (last_request.dlat, last_request.dlon)
            vehicle_location = current_vehicle.location
            distance_to_pickup = calculate_distance(vehicle_location, pickup_location)
            wait_time = calculate_wait_time(distance_to_pickup)
            effective_distance = sum(current_request.trip_distance)
            actual_distance, trip_time = calculate_actual_trip_time(sorted_request)
            current_vehicle.start_service(dropoff_location, wait_time, trip_time, actual_distance, effective_distance, passenger_count_in_vehicle)
            total_waiting_time += wait_time

        return total_waiting_time, total_assigned_vehicles, total_passenger_count, zero_hop_requests_count


    def dispatch(self, actions):
        vehicle_locations, target_locations = prepare_vehicle_and_target_locations(self, actions)
        distances = calculate_distances(self, vehicle_locations, target_locations)
        feature_matrix = prepare_feature_matrix(self, vehicle_locations, target_locations, distances)
        trip_times = self.eta_model.predict(feature_matrix)
        execute_actions(self, actions, trip_times)


    def get_vehicles_location(self):
        vehicles = get_vehicle_locations_from_dataframe(self.vehicles)
        return vehicles

