import math
import geohash2 as Geohash
from hyperparameters import *

"""
Status     free     temporal   
WAITING:    1        real        
MOVING:     1        real        
SERVING:    0        future  
"""


class Vehicle(object):
    def __init__(self, vehicle_id, location):
        self.id = vehicle_id
        self.status = 'WAITING'
        self.location = location
        self.zone = Geohash.encode(location[0], location[1], precision=GEOHASH_PRECISION)
        self.available = True
        self.trajectory = []
        self.eta = 0
        self.idle = 0
        self.total_idle = 0
        self.total_service = 0
        self.reward = 0
        self.effective_d =0
        self.actual_d = 0

    def __init__(self, vehicle_identifier, vehicle_location):
        self.id = vehicle_identifier
        self.status = 'WAITING'
        self.location = vehicle_location
        self.zone = Geohash.encode(vehicle_location[0], vehicle_location[1], precision=GEOHASH_PRECISION)
        self.available = True
        self.trajectory = []
        self.eta = 0
        self.idle = 0
        self.total_idle = 0
        self.total_service = 0
        self.reward = 0
        self.effective_d = 0
        self.actual_d = 0
        

    def update_location(self, new_location):
        latitude, longitude = new_location
        self.location = (latitude, longitude)
        self.zone = Geohash.encode(latitude, longitude, precision=GEOHASH_PRECISION)

    def transition(self):
        cost = 0
        if self.status != 'SERVING':
            self.idle += TIMESTEP/60.0
        if self.eta > 0:
            time = min(TIMESTEP/60.0, self.eta)
            self.eta -= time
            if self.status == 'MOVING':
                cost = time
                self.reward -= cost
        if self.eta <= 0:
            if self.status == 'SERVING':
                self.available = True
                self.status = 'WAITING'
            elif self.status == 'MOVING':
                self.status = 'WAITING'
        return cost

    def start_service(self, destination_location, wait_duration, trip_duration, actual_distance, effective_distance, num_passengers):
        if not self.available:
            print("The vehicle #%d is not available for service." % self.id)
            return False
        self.available = False
        self.update_location(destination_location)
        self.total_idle += self.idle + wait_duration
        num_hops = effective_distance / actual_distance
        self.idle = 0
        self.eta = wait_duration + trip_duration
        self.total_service += trip_duration
        self.reward += (math.sqrt(RIDE_REWARD * num_passengers) + TRIP_REWARD * trip_duration - math.sqrt(WAIT_COST * wait_duration) - (HOP_REWARD * num_hops))
        self.trajectory = []
        self.status = 'SV'
        self.effective_d += effective_distance
        self.actual_d += actual_distance 
        return True

    def route(self, path, trip_time):
        if not self.available:
            print ("The vehicle #%d is not available for service." % self.id)
            return False
        self.eta = trip_time
        self.trajectory = path
        self.status = 'MOVING'
        return True


    def get_state(self):
        if self.trajectory:
            latitude, longitude = self.trajectory[-1]
            destination_zone = Geohash.encode(latitude, longitude, precision=GEOHASH_PRECISION)
        else:
            destination_zone = self.zone
        latitude, longitude = self.location
        return (self.id, int(self.available), self.zone, destination_zone,
                self.eta, self.status, self.reward, latitude, longitude, self.idle,self.effective_d,self.actual_d)

    def get_location(self):
        latitude, longitude = self.location
        return (self.id, latitude, longitude, int(self.available))


    def get_score(self):
        return (self.id, self.reward, self.total_service, self.total_idle)