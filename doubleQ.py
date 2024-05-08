import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import sys
import os
from collections import deque
from keras.models import Model
from keras.layers import Input, Flatten, concatenate, Convolution2D, \
    AveragePooling2D, MaxPooling2D, Cropping2D, Lambda, Multiply, concatenate


DEEP_LEARNING_ENGINE = 'tensorflow'
DATA_DIRECTORY = 'data/'
ENVIRONMENT_NAME = 'duel'
NUM_OF_AGGREGATION = 5
LATITUDE_INCREMENT = 0.3 / 218 * NUM_OF_AGGREGATION
LONGITUDE_INCREMENT = 0.3 / 218 * NUM_OF_AGGREGATION
MIN_LATITUDE = 40.6003
MAX_LATITUDE = 40.9003
MIN_LONGITUDE = -74.0407
MAX_LONGITUDE = -73.7501
X_SCALING_FACTOR = 100.0
W_SCALING_FACTOR = 100.0

MAP_WIDTH_SIZE = int((MAX_LONGITUDE - MIN_LONGITUDE) / LONGITUDE_INCREMENT) + 1
MAP_HEIGHT_SIZE = int((MAX_LATITUDE - MIN_LATITUDE) / LATITUDE_INCREMENT) + 1
MAIN_PATH_LENGTH = 51
MAIN_PATH_DEPTH = 5
AUXILIARY_PATH_LENGTH = 15 #23
AUXILIARY_PATH_DEPTH = 11
MAXIMUM_MOVEMENT = 7
OUTPUT_LENGTH_SIZE = 15
STAY_ACTION_INDEX = OUTPUT_LENGTH_SIZE * OUTPUT_LENGTH_SIZE / 2

DISCOUNT_RATE = 0.9
EXPLORATION_STEP_LIMIT = 500
INITIAL_EXPLORATION_RATE = 0.10
FINAL_EXPLORATION_RATE = 0.05
INITIAL_TEMPERATURE = 0.10
FINAL_TEMPERATURE = 0.0
INITIAL_REPLAY_MEMORY_SIZE = 0
REPLAY_MEMORY_SIZE_LIMIT = 10000
SAVE_FREQUENCY = 1000
BATCH_SIZE_VALUE = 64
NUM_BATCHES = 2
SAMPLES_PER_FRAME = 2
TARGET_NETWORK_UPDATE_INTERVAL = 150
SUMMARY_UPDATE_INTERVAL = 60
LEARNING_RATE_VALUE = 0.00025
MOMENTUM_RATE = 0.95
MIN_GRADIENT = 0.01
NETWORK_SAVING_DIRECTORY = DATA_DIRECTORY + '/saved_networks'
SUMMARY_SAVING_DIRECTORY = DATA_DIRECTORY + '/summary'
MODEL_PATH = 'eta_model/model.h5'


def pad_array(array, size):
    return np.pad(array, int((size - 1) / 2), 'constant')

def crop_array(array, x, y, size):
    return array[x:x + size, y:y + size]

def pad_crop(F, x, y, size):
    padded_F = pad_array(F, size)
    return crop_array(padded_F, x, y, size)


def build_input(shape):
    return Input(shape=shape, dtype='float32')

def convolution_layer(x, filters, kernel_size, activation, border_mode):
    return Convolution2D(filters, kernel_size, activation=activation, border_mode=border_mode, data_format="channels_first")(x)

def build_model(input_layer, output_layer):
    return Model(input=input_layer, output=output_layer)

def build_d_network():
    input_layer = build_input((6, 212, 219))

    x = convolution_layer(input_layer, 8, (5, 5), 'relu', 'same')
    x = convolution_layer(x, 16, (3, 3), 'relu', 'same')
    output_layer = convolution_layer(x, 1, (1, 1), 'relu', 'same')

    model = build_model(input_layer, output_layer)
    
    return model

def build_main_input(shape):
    return Input(shape=shape, dtype='float32')

def build_aux_input(shape):
    return Input(shape=shape, dtype='float32')

def slice_main_input(main_input):
    return Lambda(lambda x: x[:, :-1, :, :])(main_input)

def pool_main_input(sliced_input):
    return MaxPooling2D(pool_size=(OUTPUT_LENGTH_SIZE, OUTPUT_LENGTH_SIZE), strides=(1, 1), data_format="channels_first")(sliced_input)

def crop_ave_input(ave, c):
    e = int(c)
    return Cropping2D(cropping=((e, e), (e, e)), data_format="channels_first")(ave)

def pool_main_input_ave(sliced_input):
    return AveragePooling2D(pool_size=(OUTPUT_LENGTH_SIZE, OUTPUT_LENGTH_SIZE), strides=(1, 1), data_format="channels_first")(sliced_input)

def crop_gra_test(sliced_input, c):
    e = int(c)
    return Cropping2D(cropping=((e * 2, e * 2), (e * 2, e * 2)), data_format="channels_first")(sliced_input)

def merge_layers(gra_test, ave1, ave2):
    return concatenate([gra_test, ave1, ave2], axis=1)

def convolution_layer(x, filters, kernel_size, activation, name):
    return Convolution2D(filters, kernel_size, activation=activation, name=name, data_format="channels_first")(x)

def flatten_layer(x):
    return Flatten()(x)

def build_model(main_input, aux_input, q_values_test1):
    return Model([main_input, aux_input], q_values_test1)

def build_q_network():
    main_input = build_main_input((MAIN_PATH_DEPTH, MAIN_PATH_LENGTH, MAIN_PATH_LENGTH))
    aux_input = build_aux_input((AUXILIARY_PATH_DEPTH, AUXILIARY_PATH_LENGTH, AUXILIARY_PATH_LENGTH))
    c = OUTPUT_LENGTH_SIZE / 2

    sliced_input = slice_main_input(main_input)
    ave = pool_main_input(sliced_input)
    ave1 = crop_ave_input(ave, c)
    ave2 = pool_main_input_ave(sliced_input)
    gra_test = crop_gra_test(sliced_input, c)

    merged_layer = merge_layers(gra_test, ave1, ave2)

    x = convolution_layer(merged_layer, 16, (5, 5), 'relu', 'main/conv_1')
    x = convolution_layer(x, 32, (3, 3), 'relu', 'main/conv_2')
    main_output = convolution_layer(x, 64, (3, 3), 'relu', 'main/conv_3')

    aux_output = convolution_layer(aux_input, 16, (1, 1), 'relu', 'ayx/conv')

    merged_output = concatenate([main_output, aux_output], axis=1)
    x = convolution_layer(merged_output, 128, (1, 1), 'relu', 'merge/conv')
    x = convolution_layer(x, 1, (1, 1), None, 'main/q_value')

    z = flatten_layer(x)
    legal = flatten_layer(Lambda(lambda x: x[:, -1:, :, :])(aux_input))
    q_values_test1 = Multiply()([z, legal])

    model = build_model(main_input, aux_input, q_values_test1)

    return main_input, aux_input, q_values_test1, model


class Agent(object):
    def __init__(self, geohash_table, time_step, cycle, demand_cycle, training=True, load_network=False):
        self.initialize_variables(geohash_table, time_step, cycle, demand_cycle)
        self.initialize_geohash_table()
        self.create_action_space()
        self.build_q_network()
        if training:
            self.initialize_training()
        else:
            self.initialize_inference()

    def initialize_variables(self, geohash_table, time_step, cycle, demand_cycle):
        self.geo_table = geohash_table
        self.time_step = time_step
        self.cycle = cycle
        self.training = training
        self.demand_cycle = demand_cycle
        self.x_matrix = np.zeros((AUXILIARY_PATH_LENGTH, AUXILIARY_PATH_LENGTH))
        self.y_matrix = np.zeros((AUXILIARY_PATH_LENGTH, AUXILIARY_PATH_LENGTH))
        self.d_matrix = np.zeros((AUXILIARY_PATH_LENGTH, AUXILIARY_PATH_LENGTH))

    def initialize_geohash_table(self):
        self.geo_table['x'] = np.uint8((self.geo_table.lon - MIN_LONGITUDE) / LONGITUDE_INCREMENT)
        self.geo_table['y'] = np.uint8((self.geo_table.lat - MIN_LATITUDE) / LATITUDE_INCREMENT)
        self.create_xy_to_geohash_mapping()
        self.create_legal_map()

    def create_xy_to_geohash_mapping(self):
        self.xy2g = [[list(self.geo_table[(self.geo_table.x == x) & (self.geo_table.y == y)].index)
                      for y in range(MAP_HEIGHT_SIZE)] for x in range(MAP_WIDTH_SIZE)]

    def create_legal_map(self):
        self.legal_map = np.zeros((MAP_WIDTH_SIZE, MAP_HEIGHT_SIZE))
        for x in range(MAP_WIDTH_SIZE):
            for y in range(MAP_HEIGHT_SIZE):
                if self.xy2g[x][y]:
                    self.legal_map[x, y] = 1

    def create_action_space(self):
        self.action_space = [(x, y) for x in range(-MAXIMUM_MOVEMENT, MAXIMUM_MOVEMENT + 1) for y in range(-MAXIMUM_MOVEMENT, MAXIMUM_MOVEMENT + 1)]
        self.num_actions = len(self.action_space)

    def build_q_network(self):
        self.s, self.x, self.q_values, q_network = build_q_network()
        self.q_network_weights = q_network.trainable_weights

    def initialize_training(self):
        self.build_target_network()
        self.build_training_operation()
        self.setup_summary()
        self.initialize_replay_memory()
        self.initialize_epsilon_beta()

    def build_target_network(self):
        self.st, self.xt, self.target_q_values, target_network = build_q_network()
        target_network_weights = target_network.trainable_weights
        self.update_target_network = [target_network_weights[i].assign(self.q_network_weights[i]) for i in
                                      range(len(target_network_weights))]

    def build_training_operation(self):
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(self.q_network_weights)

    def setup_summary(self):
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SUMMARY_SAVING_DIRECTORY, self.sess.graph)

    def initialize_replay_memory(self):
        self.num_iters = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.state_buffer = deque()
        self.replay_memory = deque()
        self.replay_memory_weights = deque()

    def initialize_epsilon_beta(self):
        self.epsilon = INITIAL_EXPLORATION_RATE
        self.epsilon_step = (FINAL_EXPLORATION_RATE - INITIAL_EXPLORATION_RATE) / EXPLORATION_STEP_LIMIT
        self.beta = INITIAL_TEMPERATURE
        self.beta_step = (FINAL_TEMPERATURE - INITIAL_TEMPERATURE) / EXPLORATION_STEP_LIMIT
        self.num_iters -= INITIAL_REPLAY_MEMORY_SIZE
        self.start_iter = self.num_iters

    def initialize_inference(self):
        self.demand_model = build_d_network()
        self.demand_model.load_weights(MODEL_PATH)

    def load_network(self):
        if not os.path.exists(NETWORK_SAVING_DIRECTORY):
            os.makedirs(NETWORK_SAVING_DIRECTORY)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(NETWORK_SAVING_DIRECTORY))




    def reset(self, requests, dayofweek, minofday):
        self.dayofweek = dayofweek
        self.minofday = minofday
        self.initialize_request_buffer(requests)
        self.initialize_geo_table()
        self.initialize_state_buffer()
        self.start_iter = self.num_iters
        self.total_q_max = 0
        self.total_loss = 0

    def initialize_request_buffer(self, requests):
        self.request_buffer = deque()
        self.calculate_request_count(requests)

    def calculate_request_count(self, requests):
        self.geo_table['W_1'] = 0
        self.geo_table['W_2'] = 0
        minutes = (requests.second.values[-1] - requests.second.values[0]) / 60.0
        count = requests.groupby('phash')['plat'].count() * self.time_step / minutes
        for _ in range(int(60 / self.time_step)):
            self.request_buffer.append(count.copy())

    def initialize_geo_table(self):
        self.geo_table['W_1'] = 0
        self.geo_table['W_2'] = 0

    def initialize_state_buffer(self):
        self.state_buffer = deque()


    def init_train(self, N, init_memory, summary_duration=5):
        file_path = "epoch_qmax_loss.txt"
        self.initialize_replay_memory(init_memory)
        with open(file_path, 'w') as file:
            for i in range(N):
                self.update_target_network_periodically(i)
                self.record_summary_stats(file, i, summary_duration)
                self.train_network()

    def initialize_replay_memory(self, init_memory):
        self.replay_memory = deque(init_memory)
        self.replay_memory_weights = deque([len(m[3]) for m in init_memory])

    def update_target_network_periodically(self, iteration):
        if iteration % TARGET_NETWORK_UPDATE_INTERVAL == 0:
            self.sess.run(self.update_target_network)

    def record_summary_stats(self, file, iteration, summary_duration):
        if iteration % summary_duration == 0:
            avg_q_max = self.total_q_max / summary_duration
            avg_loss = self.total_loss / summary_duration
            output_file_line = 'ITER: {:d} / Q_MAX: {:.3f} / LOSS: {:.3f}'.format(iteration, avg_q_max, avg_loss)
            print(output_file_line)
            file.write(output_file_line)
            self.total_q_max = 0
            self.total_loss = 0

    def get_actions(self, vehicles, requests):
        self.update_time()
        if not self.training:
            self.update_demand(requests)
        env_state, resource = self.preprocess(vehicles)

        if self.training:
            self.execute_training_actions(env_state)
        else:
            self.execute_inference_actions(env_state, resource)

        actions = self.select_actions(env_state, resource)
        self.num_iters += 1
        return actions

    def execute_training_actions(self, env_state):
        self.memorize_experience(env_state, vehicles)
        if self.num_iters >= 0:
            self.update_target_and_train_network()

    def execute_inference_actions(self, env_state, resource):
        if len(resource.index) > 0:
            actions = self.e_greedy(env_state, resource)
        else:
            actions = []

    def update_target_and_train_network(self):
        if self.num_iters % TARGET_NETWORK_UPDATE_INTERVAL == 0:
            self.sess.run(self.update_target_network)
        self.train_network()
        if self.num_iters % SUMMARY_UPDATE_INTERVAL == 0:
            self.write_summary()
        if self.num_iters % SAVE_FREQUENCY == 0:
            self.save_network()

    def select_actions(self, env_state, resource):
        if self.training:
            return self.e_greedy(env_state, resource)
        else:
            return self.run_policy(env_state, resource)

    def save_network(self):
        save_path = self.saver.save(self.sess, NETWORK_SAVING_DIRECTORY + '/' + ENVIRONMENT_NAME,
                                    global_step=self.num_iters)
        print('Successfully saved: ' + save_path)



    def update_time(self):
        self.increment_time()
        self.adjust_day_of_week()

    def increment_time(self):
        self.minofday += self.time_step
        if self.minofday >= MINUTES_PER_DAY:
            self.minofday -= MINUTES_PER_DAY

    def adjust_day_of_week(self):
        if self.minofday == 0:
            self.dayofweek = (self.dayofweek + 1) % DAYS_PER_WEEK


    def update_demand(self, requests):
        self.update_request_buffer(requests)
        if self.should_update_demand():
            self.update_geo_table_with_weights()
            self.calculate_auxiliary_features()
            self.predict_demand_and_update_geo_table()

    def update_request_buffer(self, requests):
        if len(self.request_buffer) >= REQUEST_BUFFER_SIZE:
            self.request_buffer.popleft()
        count = requests.groupby('phash')['plat'].count()
        self.request_buffer.append(count)

    def should_update_demand(self):
        return self.num_iters % 10 == 0

    def update_geo_table_with_weights(self):
        self.geo_table[['W_1', 'W_2']] = 0
        for i, W in enumerate(self.request_buffer):
            if i < REQUEST_BUFFER_SIZE / 2:
                self.geo_table.loc[W.index, 'W_1'] += W.values
            else:
                self.geo_table.loc[W.index, 'W_2'] += W.values

    def calculate_auxiliary_features(self):
        min_percentage = self.minofday / MINUTES_PER_DAY
        day_percentage = self.dayofweek / DAYS_PER_WEEK
        self.auxiliary_features = [np.sin(min_percentage), np.cos(min_percentage), np.sin(day_percentage), np.cos(day_percentage)]

    def predict_demand_and_update_geo_table(self):
        demand = self.predict_demand()
        self.geo_table['W'] = demand[self.geo_table.x_.values, self.geo_table.y_.values]

    def predict_demand(self):
        W_1 = self.geo_table.pivot(index='x_', columns='y_', values='W_1').fillna(0).values
        W_2 = self.geo_table.pivot(index='x_', columns='y_', values='W_2').fillna(0).values
        input_features = [[W_1, W_2] + [np.ones(W_1.shape) * x for x in self.auxiliary_features]]
        return self.demand_model.predict(np.float32(input_features))[0, 0]


    def preprocess(self, vehicles):
        self.calculate_vehicle_positions(vehicles)
        self.calculate_available_vehicles(vehicles)
        self.calculate_idle_vehicles(vehicles)

        self.calculate_geo_table_metrics()
        self.calculate_dataframe_metrics()
        env_state = self.create_environment_state()
        return env_state, self.R_idle

    def calculate_vehicle_positions(self, vehicles):
        vehicles['x'] = np.uint8((vehicles.lon - MIN_LONGITUDE) / LONGITUDE_INCREMENT)
        vehicles['y'] = np.uint8((vehicles.lat - MIN_LATITUDE) / LATITUDE_INCREMENT)

    def calculate_available_vehicles(self, vehicles):
        self.R = vehicles[vehicles.available == 1]

    def calculate_idle_vehicles(self, vehicles):
        self.R_idle = self.R[self.R.idle % self.cycle == 0]

    def calculate_geo_table_metrics(self):
        self.geo_table['X'] = self.R.groupby('dest_geohash')['available'].count()
        self.geo_table.fillna(0, inplace=True)
        self.geo_table['ratio'] = (self.geo_table.X / (self.geo_table.X.sum() + 1)) - (self.geo_table.W / (self.geo_table.W.sum() + 1))

    def calculate_dataframe_metrics(self):
        self.df['W'] = self.geo_table.groupby(['x', 'y'])['W'].sum()
        self.df['X'] = self.R.groupby(['x', 'y'])['available'].count()
        self.df['X1'] = self.R1.groupby(['x', 'y'])['available'].count()
        self.df['X2'] = self.R2.groupby(['x', 'y'])['available'].count()
        self.df['X_idle'] = self.R_idle.groupby(['x', 'y'])['available'].count()
        self.df.fillna(0, inplace=True)
        self.df['X1'] -= self.df.W / 2.0
        self.df['X2'] -= self.df.W

    def create_environment_state(self):
        df = self.df.reset_index()
        W = df.pivot(index='x', columns='y', values='W').fillna(0).values.astype(np.float32) / W_SCALING_FACTOR
        X = df.pivot(index='x', columns='y', values='X').fillna(0).values.astype(np.float32) / X_SCALING_FACTOR
        X1 = df.pivot(index='x', columns='y', values='X1').fillna(0).values.astype(np.float32) / X_SCALING_FACTOR
        X2 = df.pivot(index='x', columns='y', values='X2').fillna(0).values.astype(np.float32) / X_SCALING_FACTOR
        X_idle = df.pivot(index='x', columns='y', values='X_idle').fillna(0).values.astype(np.float32) / X_SCALING_FACTOR
        env_state = [W, X, X1, X2, X_idle]
        return env_state


    def e_greedy(self, env_state, resource):
        dispatch = []
        actions = []
        xy_idle = self.get_idle_positions(env_state)

        if self.should_explore():
            aids = self.explore_actions(xy_idle, env_state)
        else:
            aids = self.exploit_actions(env_state)

        for vid, (x, y) in resource[['x', 'y']].iterrows():
            aid, action = self.select_action(aids, xy_idle, x, y)
            dispatch.append((vid, action))
            actions.append(aid)

        self.update_state_buffer(resource, actions, env_state)
        return dispatch

    def should_explore(self):
        return self.epsilon < 1

    def get_idle_positions(self, env_state):
        return [(x, y) for y in range(MAP_HEIGHT_SIZE) for x in range(MAP_WIDTH_SIZE) if env_state[-1][x, y] > 0]

    def explore_actions(self, xy_idle, env_state):
        xy2index = {(x, y): i for i, (x, y) in enumerate(xy_idle)}
        aux_features = np.float32(self.create_aux_feature(self.minofday, self.dayofweek, xy_idle))
        main_features = np.float32(self.create_main_feature(env_state, xy_idle))
        aids = np.argmax(self.q_values.eval(feed_dict={
            self.s: np.float32(main_features), self.x: np.float32(aux_features)}), axis=1)
        return aids

    def exploit_actions(self, env_state):
        return [STAY_ACTION_INDEX if self.beta >= np.random.random() else np.random.randint(self.num_actions) for _ in range(len(xy_idle))]

    def select_action(self, aids, xy_idle, x, y):
        xy2index = {(x, y): i for i, (x, y) in enumerate(xy_idle)}
        if self.should_explore():
            aid = aids[xy2index[(x, y)]]
        else:
            aid = STAY_ACTION_INDEX if self.beta >= np.random.random() else np.random.randint(self.num_actions)
        action = STAY_ACTION_INDEX
        if aid != STAY_ACTION_INDEX:
            move_x, move_y = self.action_space[aid]
            x_ = x + move_x
            y_ = y + move_y
            if self.is_within_bounds(x_, y_) and self.is_legal_position(x_, y_):
                lat, lon = self.get_nearest_legal_position(x_, y_)
                dispatch.append((vid, (lat, lon)))
                action = aid
        return aid, action

    def is_within_bounds(self, x, y):
        return 0 <= x < MAP_WIDTH_SIZE and 0 <= y < MAP_HEIGHT_SIZE

    def is_legal_position(self, x, y):
        return len(self.xy2g[x][y]) > 0

    def get_nearest_legal_position(self, x, y):
        g = self.xy2g[x][y]
        gmin = self.geo_table.loc[g, 'ratio'].argmin()
        gmin_row = self.geo_table.iloc[gmin]
        return gmin_row.lat, gmin_row.lon

    def update_state_buffer(self, resource, actions, env_state):
        state_dict = {
            'minofday': self.minofday,
            'dayofweek': self.dayofweek,
            'vid': resource.index,
            'env': env_state,
            'pos': resource[['x', 'y']].values.astype(np.uint8),
            'reward': resource['reward'].values.astype(np.float32),
            'action': np.uint8(actions)
        }
        self.state_buffer.append(state_dict)


    def run_policy(self, env_state, resource):
        dispatch = []
        W, X, X1, X2, X_idle = env_state
        xy_idle = self.get_idle_positions(X_idle)
        xy2index = {(x, y): i for i, (x, y) in enumerate(xy_idle)}
        aux_features = self.create_aux_feature(self.minofday, self.dayofweek, xy_idle)

        for vid, (x, y) in resource[['x', 'y']].iterrows():
            aid = self.choose_action(env_state, aux_features, xy2index, x, y)
            new_x, new_y = self.take_action(aid, x, y)
            if (new_x, new_y) != (x, y):
                lat, lon = self.dispatch_vehicle(new_x, new_y)
                dispatch.append((vid, (lat, lon)))
            self.update_environment_state(X1, X2, X_idle, x, y, new_x, new_y)

        return dispatch

    def get_idle_positions(self, X_idle):
        return [(x, y) for y in range(MAP_HEIGHT_SIZE) for x in range(MAP_WIDTH_SIZE) if X_idle[x, y] > 0]

    def create_aux_feature(self, minofday, dayofweek, xy_idle):
        aux_features = np.float32(self.create_aux_feature(minofday, dayofweek, xy_idle))
        return aux_features[[xy2index[(x, y)]]]

    def choose_action(self, env_state, aux_features, xy2index, x, y):
        main_feature = np.float32(self.create_main_feature(env_state, [(x, y)]))
        aid = np.argmax(self.q_values.eval(feed_dict={
            self.s: np.float32(main_feature), self.x: np.float32(aux_features)}), axis=1)[0]
        return aid

    def take_action(self, aid, x, y):
        if aid == STAY_ACTION_INDEX:
            return x, y
        move_x, move_y = self.action_space[aid]
        new_x, new_y = x + move_x, y + move_y
        return new_x, new_y

    def dispatch_vehicle(self, new_x, new_y):
        g = self.xy2g[new_x][new_y]
        gmin = self.geo_table.loc[g, 'ratio'].argmin()
        lat, lon = self.geo_table.loc[gmin, ['lat', 'lon']]
        return lat, lon

    def update_environment_state(self, X1, X2, X_idle, x, y, new_x, new_y):
        X1[x, y] -= 1.0 / X_SCALING_FACTOR
        X2[x, y] -= 1.0 / X_SCALING_FACTOR
        X_idle[x, y] -= 1.0 / X_SCALING_FACTOR
        X1[new_x, new_y] += 1.0 / X_SCALING_FACTOR
        X2[new_x, new_y] += 1.0 / X_SCALING_FACTOR

    def create_main_feature(self, env_state, positions):
        features = []
        for x, y in positions:
            feature = []
            for s in env_state:
                cropped_s = pad_crop(s, x, y, MAIN_PATH_LENGTH)
                feature.append(cropped_s)
            features.append(feature)
        return features


    def create_aux_feature(self, minofday, dayofweek, positions):
        aux_features = []
        min_percentage = minofday / MINUTES_PER_DAY
        day_percentage = (dayofweek + int(min_percentage)) / DAYS_PER_WEEK

        for x, y in positions:
            aux = np.zeros((AUXILIARY_PATH_DEPTH, AUXILIARY_PATH_LENGTH, AUXILIARY_PATH_LENGTH))
            aux[0, :, :] = np.sin(min_percentage)
            aux[1, :, :] = np.cos(min_percentage)
            aux[2, :, :] = np.sin(day_percentage)
            aux[3, :, :] = np.cos(day_percentage)
            aux[4, AUXILIARY_PATH_LENGTH // 2, AUXILIARY_PATH_LENGTH // 2] = 1.0
            aux[5, :, :] = x / MAP_WIDTH_SIZE
            aux[6, :, :] = y / MAP_HEIGHT_SIZE
            aux[7, :, :] = (x + self.x_matrix) / MAP_WIDTH_SIZE
            aux[8, :, :] = (y + self.y_matrix) / MAP_HEIGHT_SIZE
            aux[9, :, :] = self.d_matrix

            legal_map = pad_crop(self.legal_map, x, y, AUXILIARY_PATH_LENGTH)
            legal_map[AUXILIARY_PATH_LENGTH // 2 + 1, AUXILIARY_PATH_LENGTH // 2 + 1] = 1
            aux[10, :, :] = legal_map

            aux_features.append(aux)

        return aux_features


    def memorize_experience(self, env_state, vehicles):
        if not self.state_buffer:
            return

        if self._is_cycle_complete():
            state_action = self.state_buffer.popleft()
            weight = len(state_action['vid'])
            if weight > 0:
                replay_data = self._prepare_replay_data(state_action, vehicles)
                self._update_replay_memory(replay_data, weight)

    def _is_cycle_complete(self):
        first_state = self.state_buffer[0]
        return (first_state['minofday'] + self.cycle) % MINUTES_PER_DAY == self.minofday

    def _prepare_replay_data(self, state_action, vehicles):
        vehicle_data = vehicles.loc[state_action['vid'], ['geohash', 'reward', 'eta', 'lat', 'lon']]

        state_action['reward'] = vehicle_data['reward'].values.astype(np.float32) - state_action['reward']
        state_action['delay'] = np.round(vehicle_data['eta'].values / self.cycle).astype(np.uint8)
        state_action['next_pos'] = self.geo_table.loc[vehicle_data['geohash'], ['x', 'y']].values.astype(np.uint8)
        state_action['next_env'] = env_state

        return [state_action[key] for key in self.replay_memory_keys]

    def _update_replay_memory(self, replay_data, weight):
        self.replay_memory.append(replay_data)
        self.replay_memory_weights.append(weight)

        if len(self.replay_memory) > REPLAY_MEMORY_SIZE_LIMIT:
            self.replay_memory.popleft()
            self.replay_memory_weights.popleft()


    def train_network(self):
        main_batch, aux_batch, action_batch, reward_batch, next_main_batch, next_aux_batch, delay_batch = self._prepare_batches()

        target_q_batch = self._calculate_target_q_batch(next_main_batch, next_aux_batch)
        a_batch = np.argmax(self.q_values.eval(feed_dict={
            self.s: np.array(next_main_batch),
            self.x: np.array(next_aux_batch)
        }), axis=1)
        target_q_max_batch = target_q_batch[range(BATCH_SIZE_VALUE * NUM_BATCHES), a_batch]
        self.total_q_max += target_q_max_batch.mean()

        y_batch = self._calculate_y_batch(reward_batch, delay_batch, target_q_max_batch)
        main_batch, aux_batch, action_batch, y_batch = self._shuffle_batches(main_batch, aux_batch, action_batch, y_batch)

        total_loss = self._update_model(main_batch, aux_batch, action_batch, y_batch)
        self.total_loss += total_loss / NUM_BATCHES

    def _prepare_batches(self):
        main_batch, aux_batch, action_batch, reward_batch, next_main_batch, next_aux_batch, delay_batch = [], [], [], [], [], [], []
        weights = np.array(self.replay_memory_weights, dtype=np.float32)
        memory_index = np.random.choice(range(len(self.replay_memory)), size=int(BATCH_SIZE_VALUE * NUM_BATCHES / SAMPLES_PER_FRAME), p=weights / weights.sum())

        for i in memory_index:
            data = self.replay_memory[i]
            samples = np.random.randint(self.replay_memory_weights[i], size=SAMPLES_PER_FRAME)
            aux_batch += self.create_aux_feature(data[0], data[1], data[3][samples])
            next_aux_batch += self.create_aux_feature(data[0] + self.cycle, data[1], data[7][samples])
            main_batch += self.create_main_feature(data[2], data[3][samples])
            next_main_batch += self.create_main_feature(data[6], data[7][samples])
            action_batch += data[4][samples].tolist()
            reward_batch += data[5][samples].tolist()
            delay_batch += data[8][samples].tolist()

        return main_batch, aux_batch, action_batch, reward_batch, next_main_batch, next_aux_batch, delay_batch

    def _calculate_target_q_batch(self, next_main_batch, next_aux_batch):
        return self.target_q_values.eval(
            feed_dict={
                self.st: np.array(next_main_batch),
                self.xt: np.array(next_aux_batch)
            })

    def _calculate_y_batch(self, reward_batch, delay_batch, target_q_max_batch):
        return np.array(reward_batch) + DISCOUNT_RATE ** (1 + np.array(delay_batch)) * target_q_max_batch

    def _shuffle_batches(self, main_batch, aux_batch, action_batch, y_batch):
        p = np.random.permutation(BATCH_SIZE_VALUE * NUM_BATCHES)
        return np.array(main_batch)[p], np.array(aux_batch)[p], np.array(action_batch)[p], y_batch[p]

    def _update_model(self, main_batch, aux_batch, action_batch, y_batch):
        batches = [(main_batch[k:k + BATCH_SIZE_VALUE], aux_batch[k:k + BATCH_SIZE_VALUE], action_batch[k:k + BATCH_SIZE_VALUE], y_batch[k:k + BATCH_SIZE_VALUE])
                for k in range(0, BATCH_SIZE_VALUE * NUM_BATCHES, BATCH_SIZE_VALUE)]

        total_loss = 0
        for s, x, a, y in batches:
            loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
                self.s: s,
                self.x: x,
                self.a: a,
                self.y: y
            })
            total_loss += loss
        return total_loss

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        loss = self.calculate_loss(a, y, q_value)

        grad_update = self.optimize_loss(loss, q_network_weights)

        return a, y, loss, grad_update

    def calculate_loss(self, a, y, q_value):
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        return tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

    def optimize_loss(self, loss, q_network_weights):
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE_VALUE, MOMENTUM_RATE=MOMENTUM_RATE, epsilon=MIN_GRADIENT)
        return optimizer.minimize(loss, var_list=q_network_weights)


    def setup_summary(self):
        avg_max_q = self.create_summary_variable(0., ENVIRONMENT_NAME + '/Average Max Q')
        avg_loss = self.create_summary_variable(0., ENVIRONMENT_NAME + '/Average Loss')
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(2)]
        update_ops = [avg_max_q.assign(summary_placeholders[0]), avg_loss.assign(summary_placeholders[1])]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def create_summary_variable(self, initial_value, name):
        summary_var = tf.Variable(initial_value)
        tf.summary.scalar(name, summary_var)
        return summary_var

    def write_summary(self):
        if self.num_iters >= 0:
            duration = float(self.num_iters - self.start_iter + 1)
            avg_q_max, avg_loss = self.calculate_average_stats(duration)
            self.update_summary_variables(avg_q_max, avg_loss)
            self.save_summary()
            self.print_debug_info(avg_q_max, avg_loss)

        self.reset_iteration_stats()

    def calculate_average_stats(self, duration):
        avg_q_max = self.total_q_max / duration
        avg_loss = self.total_loss / duration
        return avg_q_max, avg_loss

    def update_summary_variables(self, avg_q_max, avg_loss):
        self.update_summary_ops(avg_q_max, avg_loss)
        summary_str = self.sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, self.num_iters)

    def update_summary_ops(self, avg_q_max, avg_loss):
        stats = [avg_q_max, avg_loss]
        for i in range(len(stats)):
            self.sess.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
            })

    def save_summary(self):
        sys.stdout.flush()

    def print_debug_info(self, avg_q_max, avg_loss):
        print('ITER: {0:6d} / EPSILON: {1:.4f} / BETA: {2:.4f} / Q_MAX: {3:.3f} / LOSS: {4:.3f}'.format(
            self.num_iters, self.epsilon, self.beta, avg_q_max, avg_loss))

    def reset_iteration_stats(self):
        self.start_iter = self.num_iters
        self.total_q_max = 0
        self.total_loss = 0

    def load_network(self):
        checkpoint = self.get_checkpoint_state()
        if checkpoint and checkpoint.model_checkpoint_path:
            self.restore_checkpoint(checkpoint.model_checkpoint_path)
        else:
            self.print_new_network_message()

    def get_checkpoint_state(self):
        return tf.train.get_checkpoint_state(NETWORK_SAVING_DIRECTORY)

    def restore_checkpoint(self, model_checkpoint_path):
        self.saver.restore(self.sess, model_checkpoint_path)
        print('Successfully loaded: ' + model_checkpoint_path)

    def print_new_network_message(self):
        print('Training new network...')

    def update_future_demand(self, requests):
        self.reset_geo_table_demand()
        self.update_demand_from_requests(requests)

    def reset_geo_table_demand(self):
        self.geo_table['W'] = 0

    def update_demand_from_requests(self, requests):
        W = self.calculate_requests_demand(requests)
        self.geo_table.loc[W.index, 'W'] += W.values

    def calculate_requests_demand(self, requests):
        return requests.groupby('phash')['plat'].count()
