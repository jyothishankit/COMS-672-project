import pandas as pd
import _pickle as pickle

from simulator_v2 import FleetSimulator
from doubleQ import Agent
from experiment import run, load_trip_chunks, describe

GRAPH_PATH = 'data/nyc_network_graph.pkl'
TRIP_PATH = 'data/hoptrips_all_v3.csv'
ETA_MODEL_PATH = 'data/triptime_predictor.pkl'
GEOHASH_TABLE_PATH = 'data/zones_hop_v2.csv'
SCORE_PATH = ''
INITIAL_MEMORY_PATH = SCORE_PATH + r'data/ex_memory_v52.pkl'
INITIAL_MEMORY = True
LOAD_NETWORK = True
NUM_TRIPS = 12000000
DURATION = 800
NUM_FLEETS = 8000
NO_OP_STEPS = 0
CYCLE = 1
ACTION_UPDATE_CYCLE = 15
DEMAND_FORECAST_INTERVAL = 30
AVERAGE_CYCLE = 30
NUM_EPISODES = 20

def main():
    print("Loading models...")
    with open(GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)
    with open(ETA_MODEL_PATH, 'rb') as f:
        eta_model = pickle.load(f)
    num_fleets = NUM_FLEETS

    geohash_table = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')

    env = FleetSimulator(G, eta_model, CYCLE, ACTION_UPDATE_CYCLE)
    agent = Agent(geohash_table, CYCLE, ACTION_UPDATE_CYCLE, DEMAND_FORECAST_INTERVAL,
                  training=True, load_network=LOAD_NETWORK)
    if INITIAL_MEMORY:
        with open(INITIAL_MEMORY_PATH, 'rb') as f:
            ex_memory = pickle.load(f)
        agent.init_train(10, ex_memory)

    trip_chunks = load_trip_chunks(TRIP_PATH, NUM_TRIPS, DURATION)[:NUM_EPISODES]
    for episode, (trips, date, dayofweek, minofday) in enumerate(trip_chunks):
        # num_fleets = int(np.sqrt(len(trips)/120000.0) * NUM_FLEETS)
        env.reset(num_fleets, trips, dayofweek, minofday)
        _, requests, _, _, _,_ = env.step()
        agent.reset(requests, env.dayofweek, env.minofday)
        num_steps = int(DURATION / CYCLE - NO_OP_STEPS)

        print("#############################################################################")
        print("EPISODE: {:d} / DATE: {:d} / DAYOFWEEK: {:d} / MINUTES: {:.1f} / VEHICLES: {:d}".format(
            episode, date, env.dayofweek, env.minofday, num_fleets
        ))
        score, _ = run(env, agent, num_steps, average_cycle=AVERAGE_CYCLE, cheat=True)
        describe(score)
        score.to_csv(SCORE_PATH + 'score_dqn' + str(episode) + '.csv')

        if episode >= 0 and episode % 2 == 0:
            #print("Saving Experience Memory: {:d}").format(episode)
            with open(SCORE_PATH + 'ex_memory_v7' + str(episode) + '.pkl', 'wb') as f:
                pickle.dump(agent.replay_memory, f)


if __name__ == '__main__':
    main()
