import numpy as np
import geohelper as gh
from networkx.exception import NetworkXNoPath
import networkx as nx


class PathGenerator(object):
    def __init__(self, G, cycle=60):
        self.cycle = cycle
        self.G = G

        N = len(self.G.nodes())
        self.node_lats = np.zeros(N, 'float32')
        self.node_lons = np.zeros(N, 'float32')
        self.node_ids = np.zeros(N)

        for i, (node_id, data) in enumerate(self.G.nodes(data=True)):
            self.node_lats[i] = data['lat']
            self.node_lons[i] = data['lon']
            self.node_ids[i] = node_id
    
    def map_matching_shortest_path(self, origin, destination, weight='length', noise=1e-2, maxtry=20):
        source = self.map_match_point(origin, noise, maxtry)
        target = self.map_match_point(destination, noise, maxtry)
        path, distance = self.find_shortest_path(source, target, weight, maxtry)
        return path, distance, source, target

    def map_match_point(self, point, noise, maxtry):
        mmtry = 0
        lat, lon = point
        while True:
            try:
                u, v, d = self.map_match((lat, lon))
                break
            except ValueError:
                if mmtry > maxtry:
                    raise
                mmtry += 1
                lat += np.random.uniform(-noise, noise)
                lon += np.random.uniform(-noise, noise)
        return u, v, d

    def find_shortest_path(self, source, target, weight, maxtry):
        ptry = 0
        while True:
            try:
                path, distance = self.shortest_path(source[0], target[0], weight=weight)
                break
            except NetworkXNoPath:
                if ptry > maxtry:
                    raise
                ptry += 1
                if len(nx.single_source_shortest_path_length(self.G, source[0])) < 1000:
                    self.G.remove_node(source[0])
                if len(nx.single_source_shortest_path_length(self.G, target[0])) < 1000:
                    self.G.remove_node(target[0])
        return path, distance

    def get_node_locs(self):
        return zip(self.node_lats, self.node_lons)

    def shortest_path(self, source, target, weight='length', distance=True):
        path = nx.astar_path(self.G, source, target, self.__grand_circle, weight=weight)
        if distance:
            distance = sum(self.G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))
            return path, distance
        else:
            return path

    
    def create_trajectory(self, start_lat, start_lon, bearing, total_distance, step_distance, initial_step):
        trajectory_lats = []
        trajectory_lons = []
        current_step_distance = initial_step
        while total_distance > current_step_distance:
            start_lat, start_lon = gh.end_lat_lon(start_lat, start_lon, current_step_distance, bearing)
            trajectory_lats.append(start_lat)
            trajectory_lons.append(start_lon)
            total_distance -= current_step_distance
            current_step_distance = step_distance

        current_step_distance -= total_distance
        return zip(trajectory_lats, trajectory_lons), current_step_distance
    
    def get_segments_in_order(self, initial_point, final_point):
        edge_data = self.G.get_edge_data(initial_point, final_point)
        if not edge_data or 'lat' not in edge_data:
            edge_data = self.G.get_edge_data(final_point, initial_point)
        distances = edge_data['seg_length'] + [edge_data['length']]
        lengths = [d2 - d1 for d1, d2 in zip(distances[:-1], distances[1:])]
        bearings = edge_data['bearing']
        latitudes = edge_data['lat']
        longitudes = edge_data['lon']
        if initial_point > final_point:
            bearings = [b + np.pi for b in bearings[::-1]]
            lengths = lengths[::-1]
            latitudes = latitudes[::-1]
            longitudes = longitudes[::-1]

        return latitudes, longitudes, bearings, lengths
    
    def map_match(self, location, geographical_range=0.0018):
        latitude, longitude = location
        subgraph = self.__get_subgraph(latitude, longitude, geographical_range)
        roads = subgraph.edges()
        number_of_roads = len(roads)
        if number_of_roads == 0:
            raise ValueError("""No nodes present within search.""")

        road_lengths = np.zeros(number_of_roads, 'float16')
        road_ids = np.zeros(number_of_roads)
        road_distance = np.ones((number_of_roads), 'float16') * float('inf')
        node_distance = np.zeros((number_of_roads), 'float16')

        for i, road in enumerate(roads):
            data = self.G.get_edge_data(*road)
            if 'lat' in data:
                road_lengths[i] = data['length']
                road_ids[i] = data['id']
                (_, road_distance[i], node_distance[i]) = self.__get_nearest_segment(latitude, longitude, data)
        nearest_road_index = road_distance.argmin()
        road_tuple = list(roads)
        u = int(road_tuple[nearest_road_index][0])
        v = int(road_tuple[nearest_road_index][1])
        distance = node_distance[nearest_road_index]
        if u > v:
            u, v = v, u
        return u, v, distance

    def generate_path(self, start_point, end_point, movement_step, path_list, source_info, target_info):
        if len(path_list) < 3:
            return [end_point]
        start_u, start_v, start_distance = source_info
        target_u, target_v, target_distance = target_info
        trajectory = []
        movement_unit = movement_step
        if path_list[1] == start_u or path_list[1] == start_v:
            start_node = path_list.pop(0)
        elif path_list[0] == start_u:
            start_node = start_v
        else:
            start_node = start_v
        end_node = path_list.pop(0)
        lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)

        if start_node < end_node:
            distance = start_distance
        else:
            distance = sum(lengths) - start_distance
        for lat, lon, b, length in zip(lats[:-1], lons[:-1], bearings, lengths):
            if distance > length:
                distance -= length
                continue
            if distance > 0:
                lat, lon = gh.end_lat_lon(lat, lon, distance, b)
                trajectory.append((lat, lon))
                length -= distance
                distance = 0
            locs, movement_unit = self.create_trajectory(lat, lon, b, length, movement_step, movement_unit)
            trajectory += locs

        start_node = end_node
        for end_node in path_list[:-1]:
            lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
            for lat, lon, b, length in zip(lats[:-1], lons[:-1], bearings, lengths):
                locs, movement_unit = self.create_trajectory(lat, lon, b, length, movement_step, movement_unit)
                trajectory += locs
            start_node = end_node
        end_node = path_list[-1]
        if not (start_node == target_u or start_node == target_v):
            lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
            for lat, lon, b, length in zip(lats[:-1], lons[:-1], bearings, lengths):
                locs, movement_unit = self.create_trajectory(lat, lon, b, length, movement_step, movement_unit)
                trajectory += locs
            start_node = end_node
            if start_node == target_u:
                end_node = target_v
            else:
                end_node = target_u
        lats, lons, bearings, lengths = self.get_segments_in_order(start_node, end_node)
        if start_node < end_node:
            distance = target_distance
        else:
            distance = sum(lengths) - target_distance
        for lat, lon, b, length in zip(lats[:-1], lons[:-1], bearings, lengths):
            if distance < length:
                locs, movement_unit = self.create_trajectory(lat, lon, b, distance, movement_step, movement_unit)
                trajectory += locs
                trajectory.append(gh.end_lat_lon(lat, lon, distance, b))
                break
            locs, movement_unit = self.create_trajectory(lat, lon, b, length, movement_step, movement_unit)
            trajectory += locs
            distance -= length
        return trajectory

    
    def mm_convert(self, location, geographical_range=0.0018):
        u, v, distance = self.map_match(location, geographical_range)
        latitudes, longitudes, bearings, segment_lengths = self.get_segments_in_order(u, v)

        for lat, lon, bearing, length in zip(latitudes[:-1], longitudes[:-1], bearings, segment_lengths):
            if distance > length:
                distance -= length
            elif distance > 0:
                return gh.end_lat_lon(lat, lon, distance, bearing)
            else:
                return lat, lon

        return latitudes[-1], longitudes[-1]


    def __get_nearest_segment(self, latitude, longitude, data):
        road_latitudes = np.array(data['lat'])
        road_longitudes = np.array(data['lon'])
        bearings = np.array(data['bearing'])
        segment_lengths = np.array(data['seg_length'] + [data['length']])
        h = gh.distance_in_meters(road_latitudes, road_longitudes, latitude, longitude)
        h1 = h[:-1]
        h2 = h[1:]
        theta = gh.bearing_in_radians(road_latitudes, road_longitudes, latitude, longitude)
        cos1 = np.cos(theta[:-1] - bearings)
        cos2 = -np.cos(theta[1:] - bearings)
        d = h1 * np.sqrt(1 - cos1 ** 2) * (np.sign(cos1) == np.sign(cos2)) \
            + h1 * (np.sign(cos1) < np.sign(cos2)) + h2 * (np.sign(cos1) > np.sign(cos2))
        nearest_segment_index = d.argmin()   #size: T
        cos1 = cos1[nearest_segment_index]
        cos2 = cos2[nearest_segment_index]
        h1 = h1[nearest_segment_index]
        road_distance = d[nearest_segment_index]
        node_distance = (h1 * cos1) * (np.sign(cos1) == np.sign(cos2)) \
                            + segment_lengths[nearest_segment_index] * ~(np.sign(cos1) > np.sign(cos2)) \
                            + segment_lengths[nearest_segment_index + 1] * (np.sign(cos1) > np.sign(cos2))
        return (nearest_segment_index, road_distance, node_distance)


    def __grand_circle(self, source_id, target_id):
        source = self.G.node[source_id]
        target = self.G.node[target_id]
        return gh.distance_in_meters(source['lat'], source['lon'], target['lat'], target['lon'])
    
    def __get_subgraph(self, lat, lon, geo_range):
        latitude_min, latitude_max = lat - geo_range, lat + geo_range
        longitude_min, longitude_max = lon - geo_range, lon + geo_range
        latitudes = self.node_lats
        longitudes = self.node_lons
        sub_ids = self.node_ids[(latitudes < latitude_max) & (latitudes > latitude_min) & (longitudes < longitude_max) & (longitudes > longitude_min)]
        return self.G.subgraph(sub_ids)

