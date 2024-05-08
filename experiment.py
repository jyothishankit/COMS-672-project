import sys
import time
import pandas as pd
from random import shuffle


def initialize_score_dataframe():
    return pd.DataFrame(columns=['dayofweek', 'minofday', 'requests', 'wait_time',
                                 'reject', 'idle_trip', 'resource', 'dispatch', 'reward',
                                 'effective_dist','actual_dist' ,'agent_time','original_requests'])

def initialize_environment(env, agent, no_op_steps):
    vehicles, requests, _, _, _, _ = env.step()
    for _ in range(no_op_steps - 2):
        _, requests_, _, _, _, _ = env.step()
        requests = requests.append(requests_)
    if agent:
        agent.reset(requests, env.dayofweek, env.minofday)
    vehicles, requests, _, _, _, _ = env.step()
    return vehicles, requests

def update_future_demand(env, agent, t, num_steps, cheat_cycle):
    if t % cheat_cycle == 0:
        if t > num_steps - 30:
            return None
        return env.get_requests(num_steps=30)
    return None

def run(env, agent, num_steps, no_op_steps=2, average_cycle=1, cheat=False, cheat_cycle=15):
    score = initialize_score_dataframe()
    vehicles, requests = initialize_environment(env, agent, no_op_steps)

    start = time.time()
    prev_reward = 0
    prev_eff_dist = 0
    prev_act_dist = 0
    N = len(vehicles)

    for t in range(num_steps):
        future_requests = update_future_demand(env, agent, t, num_steps, cheat_cycle)
        if future_requests:
            agent.update_future_demand(future_requests)

        agent_start = time.time()
        actions = agent.get_actions(vehicles, requests) if agent else []

        agent_time = time.time() - agent_start
        dispatch = len(actions)
        dayofweek = env.dayofweek
        minofday = env.minofday
        vehicles, requests, wait, reject, idle, original_requests = env.step(actions)
        avg_reward = vehicles.reward.mean()
        eff_dist_total = vehicles.eff_dist.sum()
        act_dist_total = vehicles.act_dist.sum()
        
        score.loc[t] = (dayofweek, minofday, len(requests), wait, reject, idle,
                        sum(vehicles.available), dispatch, avg_reward - prev_reward, eff_dist_total - prev_eff_dist,
                        act_dist_total - prev_act_dist, agent_time, original_requests)
        prev_reward = avg_reward
        prev_eff_dist = eff_dist_total
        prev_act_dist = act_dist_total

        if t > 0 and t % average_cycle == 0:
            elapsed = time.time() - start
            W, wait, reject, dispatch, reward, effective_dist, actual_dist = score.loc[
                t - average_cycle:t - 1, ['requests', 'wait_time', 'reject', 'dispatch', 'reward', 'effective_dist',
                                           'actual_dist']].sum()
            print("t = {:d} ({:.0f} elapsed) // REQ: {:.0f} / REJ: {:.0f} / WAIT: {:.0f}/ WRT: {:.1f} / DSP: {:.2f} / RWD: {:.1f}/ED: {:.1f}/AD: {:.1f}".format(
                int(t * env.cycle), elapsed, W, reject, wait, (W - reject), dispatch / N, reward, effective_dist,
                actual_dist
            ))
            sys.stdout.flush()

    return score, env.get_vehicles_score()


def read_trip_columns(trip_path):
    return pd.read_csv(trip_path, nrows=1).columns

def read_trips(trip_path, trip_cols, sample_size, skiprows):
    return pd.read_csv(trip_path, names=trip_cols, nrows=sample_size, skiprows=skiprows+1)

def adjust_trip_times(trips):
    trips['second'] -= trips.loc[0, 'second']
    return trips

def calculate_trip_duration(trips):
    return int(trips.second.values[-1] / 60)

def extract_time_info(trips):
    dayofweek = trips.loc[0, 'dayofweek']
    minofday = trips.loc[0, 'hour'] * 60 + trips.loc[0, 'minute']
    return dayofweek, minofday

def select_trip_features(trips):
    features = ['trip_time', 'phash', 'plat', 'plon', 'dhash', 'dlat', 'dlon', 'second','trip_distance','hop_flag']
    return trips[features]

def load_trips(trip_path, sample_size, skiprows=0):
    trip_cols = read_trip_columns(trip_path)
    trips = read_trips(trip_path, trip_cols, sample_size, skiprows)
    trips = adjust_trip_times(trips)
    duration = calculate_trip_duration(trips)
    dayofweek, minofday = extract_time_info(trips)
    trips = select_trip_features(trips)
    return trips, dayofweek, minofday, duration


def calculate_num_chunks(minutes, duration):
    return int(minutes / duration)

def adjust_trip_times(trips):
    trips['second'] -= trips.second.values[0]
    return trips

def extract_chunk(trips, duration, offset, date, dayofweek, minofday):
    chunk = trips[trips.second < (duration + offset) * 60.0]
    return chunk, date, dayofweek, minofday

def remove_chunk_from_trips(trips, duration, offset):
    return trips[trips.second >= (duration + offset) * 60.0]

def update_time(dayofweek, minofday, duration):
    minofday += duration
    if minofday >= 1440:  # 24 hour * 60 minute
        minofday -= 1440
        dayofweek = (dayofweek + 1) % 7
        date += 1
    return dayofweek, minofday

def load_trip_chunks(trip_path, num_trips, duration, offset=0, randomize=True):
    trips, dayofweek, minofday, minutes = load_trips(trip_path, num_trips)
    num_chunks = calculate_num_chunks(minutes, duration)
    chunks = []
    date = 1
    for _ in range(num_chunks):
        trips = adjust_trip_times(trips)
        chunk, date, dayofweek, minofday = extract_chunk(trips, duration, offset, date, dayofweek, minofday)
        chunks.append((chunk, date, dayofweek, minofday))
        trips = remove_chunk_from_trips(trips, duration, offset)
        dayofweek, minofday = update_time(dayofweek, minofday, duration)

    if randomize:
        shuffle(chunks)

    return chunks


def adjust_trip_times(trips):
    trips['second'] -= trips.second.values[0]
    return trips

def extract_day_chunk(trips, no_op_steps):
    day_chunk = trips[trips.second < (24 * 60 + no_op_steps) * 60.0]
    return day_chunk

def remove_day_chunk_from_trips(trips):
    return trips[trips.second >= 24 * 60 * 60.0]

def update_day_time(dayofweek, date):
    dayofweek = (dayofweek + 1) % 7
    date += 1
    return dayofweek, date

def load_trip_eval(trip_path, num_trips, day_start=4, no_op_steps=30):
    trips, dayofweek, minofday, minutes = load_trips(trip_path, num_trips)
    chunks = []
    day_shift = (7 - dayofweek) % 7
    trips = trips[trips.second >= ((day_shift * 24 + day_start) * 60 - no_op_steps) * 60]
    dayofweek = 0
    minofday = day_start * 60 - no_op_steps
    date = 1 + day_shift

    while len(trips):
        trips = adjust_trip_times(trips)
        day_chunk = extract_day_chunk(trips, no_op_steps)
        chunks.append((day_chunk, date, dayofweek, minofday))
        trips = remove_day_chunk_from_trips(trips)
        dayofweek, date = update_day_time(dayofweek, date)
        if dayofweek == 0:
            break

    return chunks

def calculate_summary_stats(score):
    total_requests = int(score.requests.sum())
    total_wait = score.wait_time.sum()
    total_reject = int(score.reject.sum())
    total_idle = int(score.idle_trip.sum())
    total_reward = score.reward.sum()
    avg_num_transitions = (total_requests - total_reject) / (score.original_requests.sum())
    distance_ratio = score.effective_dist.sum() / score.actual_dist.sum()
    total_time = score.actual_dist.sum() / (score.original_requests.sum() * 15)
    
    return total_requests, total_wait, total_reject, total_idle, total_reward, avg_num_transitions, distance_ratio, total_time

def print_summary(total_requests, total_reject, total_idle, total_reward, distance_ratio, avg_num_transitions, score, total_wait, total_time):
    avg_wait = total_wait / (total_requests - total_reject)
    reject_rate = float(total_reject) / total_requests
    effort = float(total_idle) / (total_requests * 0.2 - total_reject)
    avg_time = score.agent_time.mean()

    print("=============================== SUMMARY ===============================")
    print("REQUESTS: {0:d} / REJECTS: {1:d} / IDLE: {2:d} / REWARD: {3:.0f}/RATIO_EFF_DIST: {4:.2f}/TRANS: {5:.4f}".format(
        total_requests, total_reject, total_idle, total_reward, distance_ratio, avg_num_transitions))
    print("WAIT TIME: {0:.2f} / REJECT RATE: {1:.3f} / EFFORT: {2:.2f} / TIME: {3:.2f}/TRIP_TIME: {4:.4f}".format(
        avg_wait, reject_rate, effort, avg_time, total_time))

def describe(score):
    total_requests, total_wait, total_reject, total_idle, total_reward, avg_num_transitions, distance_ratio, total_time = calculate_summary_stats(score)
    print_summary(total_requests, total_reject, total_idle, total_reward, distance_ratio, avg_num_transitions, score, total_wait, total_time)
