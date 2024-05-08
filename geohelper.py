import numpy as np

R = 6371000

def deg_to_rad(degrees):
    return np.deg2rad(degrees)

def calculate_distance(start_lat_rad, start_lon_rad, end_lat_rad, end_lon_rad):
    x = (end_lon_rad - start_lon_rad) * np.cos(0.5 * (start_lat_rad + end_lat_rad))
    y = end_lat_rad - start_lat_rad
    return np.sqrt(x**2 + y**2)

def distance_in_meters(start_lat, start_lon, end_lat, end_lon):
    start_lat_rad = deg_to_rad(start_lat)
    start_lon_rad = deg_to_rad(start_lon)
    end_lat_rad = deg_to_rad(end_lat)
    end_lon_rad = deg_to_rad(end_lon)
    return calculate_distance(start_lat_rad, start_lon_rad, end_lat_rad, end_lon_rad)

def deg_to_rad(degrees):
    return degrees * (3.141592653589793 / 180.0)

def calculate_bearing(start_lat, start_lon, end_lat, end_lon):
    start_lat_rad = deg_to_rad(start_lat)
    start_lon_rad = deg_to_rad(start_lon)
    end_lat_rad = deg_to_rad(end_lat)
    end_lon_rad = deg_to_rad(end_lon)
    
    delta_lon = end_lon_rad - start_lon_rad
    numerator = (end_lon_rad - start_lon_rad) * (0.017453292519943295 * np.cos(end_lat_rad))
    denominator = (np.cos(start_lat_rad) * np.sin(end_lat_rad)) - (np.sin(start_lat_rad) * np.cos(end_lat_rad) * np.cos(delta_lon))
    return np.arctan2(numerator, denominator)

def bearing_in_radians(start_lat, start_lon, end_lat, end_lon):
    return calculate_bearing(start_lat, start_lon, end_lat, end_lon)
    

def deg_to_rad(degrees):
    return degrees * (3.141592653589793 / 180.0)

def rad_to_deg(radians):
    return radians * (180.0 / 3.141592653589793)

def calculate_destination_lat_lon(start_lat, start_lon, distance_in_meter, bearing):
    start_lat_rad = deg_to_rad(start_lat)
    start_lon_rad = deg_to_rad(start_lon)
    alpha = distance_in_meter / R

    lat = np.arcsin(np.sin(start_lat_rad) * np.cos(alpha) + np.cos(start_lat_rad) * np.sin(alpha) * np.cos(bearing))
    
    numerator = np.sin(bearing) * np.sin(alpha) * np.cos(start_lat_rad)
    denominator = np.cos(alpha) - np.sin(start_lat_rad) * np.sin(lat)
    lon = start_lon_rad + np.arctan2(numerator, denominator)
    
    return rad_to_deg(lat), rad_to_deg(lon)

def end_lat_lon(start_lat, start_lon, distance_in_meter, bearing):
    return calculate_destination_lat_lon(start_lat, start_lon, distance_in_meter, bearing)
