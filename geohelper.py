import geohash2 as Geohash
from collections import defaultdict
import numpy as np

RADIUS_OF_EARTH = 6371000

def degrees_to_radians(degrees):
    return np.deg2rad(degrees)

def calculate_distance(rad_lat_start, rad_lon_start, rad_lat_end, rad_lon_end):
    delta_lon = (rad_lon_end - rad_lon_start) * np.cos(0.5 * (rad_lat_start + rad_lat_end))
    delta_lat = rad_lat_end - rad_lat_start
    return np.sqrt(delta_lon**2 + delta_lat**2)

def meter_distances(lat_start, lon_start, lat_end, lon_end):
    rad_lat_start = degrees_to_radians(lat_start)
    rad_lon_start = degrees_to_radians(lon_start)
    rad_lat_end = degrees_to_radians(lat_end)
    rad_lon_end = degrees_to_radians(lon_end)
    return calculate_distance(rad_lat_start, rad_lon_start, rad_lat_end, rad_lon_end)

def degrees_to_radians(degrees):
    return degrees * (3.141592653589793 / 180.0)

def calculate_value_bearing(lat_start, lon_start, lat_end, lon_end):
    rad_lat_start = degrees_to_radians(lat_start)
    rad_lon_start = degrees_to_radians(lon_start)
    rad_lat_end = degrees_to_radians(lat_end)
    rad_lon_end = degrees_to_radians(lon_end)
    
    delta_lon = rad_lon_end - rad_lon_start
    numerator = (rad_lon_end - rad_lon_start) * (0.017453292519943295 * np.cos(rad_lat_end))
    denominator = (np.cos(rad_lat_start) * np.sin(rad_lat_end)) - (np.sin(rad_lat_start) * np.cos(rad_lat_end) * np.cos(delta_lon))
    return np.arctan2(numerator, denominator)

def value_bearing_in_radians(lat_start, lon_start, lat_end, lon_end):
    return calculate_value_bearing(lat_start, lon_start, lat_end, lon_end)

def calculate_destination_lat_lon(lat_start, lon_start, meter_distance, value_bearing):
    rad_lat_start = degrees_to_radians(lat_start)
    rad_lon_start = degrees_to_radians(lon_start)
    a_value = meter_distance / RADIUS_OF_EARTH

    tal = np.arcsin(np.sin(rad_lat_start) * np.cos(a_value) + np.cos(rad_lat_start) * np.sin(a_value) * np.cos(value_bearing))
    
    numerator = np.sin(value_bearing) * np.sin(a_value) * np.cos(rad_lat_start)
    denominator = np.cos(a_value) - np.sin(rad_lat_start) * np.sin(tal)
    lon = rad_lon_start + np.arctan2(numerator, denominator)
    
    return radians_to_degrees(tal), radians_to_degrees(lon)

def lat_lon_end(lat_start, lon_start, meter_distance, value_bearing):
    return calculate_destination_lat_lon(lat_start, lon_start, meter_distance, value_bearing)

def radians_to_degrees(radians):
    return radians * (180.0 / 3.141592653589793)
