import math
import geohash2 as Geohash
from hyperparameters import *

"""
        Status      available   location    eta         storage_id
WT:     waiting     1           real        0           0
MV:     moving      1           real        >0          0
SV:     serving     0           future      >0          0
ST:     stored      0           real        0           >0
CO:     carry-out   0           real        >0          r>0
"""


class Vehicle(object):
    def __init__(self, vehicle_id, location):
        self.id = vehicle_id
        self.status = 'WT'
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
        

    def update_location(self, location):
        lat, lon = location
        self.location = (lat, lon)
        self.zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)

    def transition(self):
        cost = 0
        if self.status != 'SV':
            self.idle += TIMESTEP/60.0
        if self.eta > 0:
            time = min(TIMESTEP/60.0, self.eta)
            self.eta -= time
            if self.status == 'MV':
                cost = time
                self.reward -= cost
        if self.eta <= 0:
            if self.status == 'SV':
                self.available = True
                self.status = 'WT'
            elif self.status == 'MV':
                self.status = 'WT'
        return cost

    def start_service(self, destination, wait_time, trip_time,actual_distance,eff_distance,num_pass):
        if not self.available:
            print ("The vehicle #%d is not available for service." % self.id)
            return False
        self.available = False
        self.update_location(destination)
        self.total_idle += self.idle + wait_time
        num_hops = eff_distance /actual_distance
        self.idle = 0
        self.eta = wait_time + trip_time
        self.total_service += trip_time
        self.reward += (math.sqrt(RIDE_REWARD* num_pass) + TRIP_REWARD * trip_time - math.sqrt(WAIT_COST * wait_time) - (HOP_REWARD*num_hops))
        self.trajectory = []
        self.status = 'SV'
        self.effective_d += eff_distance
        self.actual_d +=actual_distance 
        return True

    def route(self, path, trip_time):
        if not self.available:
            print ("The vehicle #%d is not available for service." % self.id)
            return False
        self.eta = trip_time
        self.trajectory = path
        self.status = 'MV'
        return True


    def get_state(self):
        if self.trajectory:
            lat, lon = self.trajectory[-1]
            dest_zone = Geohash.encode(lat, lon, precision=GEOHASH_PRECISION)
        else:
            dest_zone = self.zone
        lat, lon = self.location
        return (self.id, int(self.available), self.zone, dest_zone,
                self.eta, self.status, self.reward, lat, lon, self.idle,self.effective_d,self.actual_d)

    def get_location(self):
        lat, lon = self.location
        return (self.id, lat, lon, int(self.available))


    def get_score(self):
        return (self.id, self.reward, self.total_service, self.total_idle)