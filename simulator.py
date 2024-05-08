import numpy as np
import pandas as pd
from pathgenerator import PathGenerator
import geohelper as gh
from hyperparameters import *
from vehicle import *

class FleetSimulator(object):
    def __init__(self, G, eta_model, cycle, max_action_time=20):
        self.router = PathGenerator(G)
        self.max_action_time = max_action_time
        self.eta_model = eta_model
        self.cycle = cycle

    def reset(self, num_vehicles, dataset, dayofweek, minofday):
        self.requests = dataset
        init_locations = self.requests[['plat', 'plon']].values[np.arange(num_vehicles) % len(self.requests)]
        self.vehicles = [Vehicle(i, init_locations[i]) for i in range(num_vehicles)]
        self.current_time = 0
        self.minofday = minofday
        self.dayofweek = dayofweek


    def get_requests(self, num_steps, offset=0):
        requests = self.requests[(self.requests.second >= self.current_time + offset * TIMESTEP)
                                 &(self.requests.second < self.current_time + TIMESTEP * (num_steps + offset))]
        return requests

    def step(self, actions=None):
        num_steps = int(self.cycle * 60.0 / TIMESTEP)

        if actions:
            self.dispatch(actions)

        requests = self.get_requests(num_steps)
        wait, reject, gas,request_hop_zero = 0, 0, 0, 0
        for _ in range(num_steps):
            for vehicle in self.vehicles:
                gas += vehicle.transition()
            X = self.get_vehicles_location()
            W = requests[(requests.second >= self.current_time)
                                 &(requests.second < self.current_time + TIMESTEP)]
            assignments = self.match(X, W)
            wait_,num_vehicles,num_passengers,request_hop_zero_ = self.assign(assignments)
            wait += wait_
            reject += len(W) - num_passengers
            request_hop_zero += request_hop_zero_
            self.update_time()
        vehicles = self.get_vehicles_dataframe()
        return vehicles, requests, wait, reject, gas, request_hop_zero
    
    def get_vehicles_dataframe(self):
        vehicles = [vehicle.get_state() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'available', 'geohash', 'dest_geohash',
                                                   'eta', 'status', 'reward', 'lat', 'lon', 'idle','eff_dist','act_dist'])
        return vehicles

    def update_time(self):
        self.current_time += TIMESTEP
        self.minofday += int(TIMESTEP / 60.0)
        if self.minofday >= 1440:
            self.minofday -= 1440
            self.dayofweek = (self.dayofweek + 1) % 7

    def get_vehicles_score(self):
        vehicles = [vehicle.get_score() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'reward', 'service_time', 'idle_time'])
        return vehicles
    
    
    def match(self, resources, tasks):
        R = resources[resources.available == 1]
        tasks_uniq = tasks.groupby(['plat','plon']).count()
        tasks_uniq = tasks_uniq.reset_index(level=['plat','plon'])
        tasks_uniq = tasks_uniq[['plat','plon']]  
        tasks_uniq['index']=tasks_uniq.apply(lambda rows:tasks[(tasks['plat']==rows['plat'].item()) & 
                            (tasks['plon']==rows['plon'].item())].index.values  , axis=1)
        
        d = gh.distance_in_meters(R.lat.values,
                                  R.lon.values,
                                  tasks_uniq.plat.values[:, None],
                                  tasks_uniq.plon.values[:, None])
        N = min(len(tasks_uniq), len(R))
        vids = np.zeros(N, dtype=int)
        for i in range(N):
            vid = d[i].argmin()
            if d[i, vid] > REJECT_DISTANCE:
                vids[i] = -1
            else:
                vids[i] = vid
                d[:, vid] = float('inf')
        index_values = tasks_uniq.index[:N][vids >= 0]
        assignments = list(zip(tasks_uniq.loc[index_values]['index'], R['id'].iloc[vids[vids >= 0]]))
        return assignments


    def assign(self, assignments):
        num_vehicles = len(assignments)
        request_hop_zero =0 
        num_passengers = 0
        wait = 0
        for r, v in assignments:
            vehicle = self.vehicles[v]
            request = self.requests.loc[r]
            num_present_vehicle_pass = len(request) 
            request_hop_zero += sum(request['hop_flag']==0)
            num_passengers += num_present_vehicle_pass
            request_sort = request.sort_values(by=['trip_time'])
            first_row = request_sort.iloc[0]
            last_row = request_sort.iloc[-1]
            ploc = (first_row.plat, first_row.plon)
            dloc = (last_row.dlat, last_row.dlon)
            vloc = vehicle.location
            d = gh.distance_in_meters(vloc[0], vloc[1], ploc[0], ploc[1])
            wait_time = (d * 2 / 1.414) / (ASSIGNMENT_SPEED * 1000 / 60)
            eff_distance = sum(request.trip_distance)
            request_sort_copy = request_sort.copy()
            request_sort.loc[:,'index_val'] = range(len(request_sort))
            request_sort_copy.loc[:,'index_val'] = request_sort['index_val'].values -1
            request_sort_copy = request_sort_copy[['dlat','dlon','index_val']]
            request_sort_join = pd.merge(request_sort, request_sort_copy, how='left', on=None, left_on='index_val', right_on='index_val',
                                         left_index=False, right_index=False, sort=False,suffixes=('_x', '_y'), 
                                         copy=True, indicator=False,validate=None)
            
            request_join_no_na = request_sort_join.dropna()
            if len(request_join_no_na) > 0:
                request_join_no_na['dist_bw_dest'] = gh.distance_in_meters(request_join_no_na.dlat_x,request_join_no_na.dlon_x,
                                                                       request_join_no_na.dlat_y,request_join_no_na.dlon_y)
                
                actual_distance = first_row.trip_distance + (sum(request_join_no_na.dist_bw_dest))/1000
                trip_time = (actual_distance/ASSIGNMENT_SPEED)*60
            else:
                actual_distance = first_row.trip_distance
                trip_time = first_row.trip_time
            vehicle.start_service(dloc, wait_time, trip_time,actual_distance,eff_distance,num_present_vehicle_pass)
            wait += wait_time

        return wait,num_vehicles,num_passengers,request_hop_zero

    def dispatch(self, actions):
        cache = []
        distances = []
        vids, targets = zip(*actions)
        vlocs = [self.vehicles[vid].location for vid in vids]
        for vloc, tloc in zip(vlocs, targets):
            try:
                p, d, s, t = self.router.map_matching_shortest_path(vloc, tloc)
                cache.append((p, s, t))
                distances.append(d)
            except:
                start_lat = vloc[0]
                start_lon = vloc[1]
                dest_lat = tloc[0]
                dest_lon = tloc[1]
                d = gh.distance_in_meters(start_lat,start_lon, dest_lat,dest_lon)
                distances.append(d)

        N = len(vids)
        X = np.zeros((N, 7))
        X[:, 0] = self.dayofweek
        X[:, 1] = self.minofday / 60.0
        X[:, 2:4] = vlocs
        X[:, 4:6] = targets
        X[:, 6] = distances
        trip_times = self.eta_model.predict(X)

        for i, vid in enumerate(vids):
            if trip_times[i] > MIN_TRIPTIME:
                eta = min(trip_times[i], self.max_action_time)
                self.vehicles[vid].route([], eta)
        return
    
    def get_vehicles_location(self):
        vehicles = [vehicle.get_location() for vehicle in self.vehicles]
        vehicles = pd.DataFrame(vehicles, columns=['id', 'lat', 'lon', 'available'])
        return vehicles

