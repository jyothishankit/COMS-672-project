import pandas as pd
import _pickle as pickle
from constants import *
from hyperparameters import *
from simulator import FleetSimulator
from doubleQ import Agent
from experiment import run, load_trip_chunks, describe
from utils import *

def main():
    print("Loading hop zone graph with demand and ETA models...")
    graph = load_graph()
    eta_model = load_eta_model()
    number_of_fleets = NUM_FLEETS
    geohash_data = pd.read_csv(GEOHASH_TABLE_PATH, index_col='geohash')
    environment = FleetSimulator(graph, eta_model, CYCLE, ACTION_UPDATE_CYCLE)
    learning_agent = Agent(geohash_data, CYCLE, ACTION_UPDATE_CYCLE, DEMAND_FORECAST_INTERVAL,
                  training=True, load_network=LOAD_NETWORK)
    if INITIAL_MEMORY:
        with open(INITIAL_MEMORY_PATH, 'rb') as file:
            example_memory = pickle.load(file)
        learning_agent.init_train(10, example_memory)

    trip_data = load_trip_chunks(TRIP_PATH, NUM_TRIPS, DURATION)[:NUM_EPISODES]
    for episode, (trips, date, day_of_week, minute_of_day) in enumerate(trip_data):
        environment.reset(number_of_fleets, trips, day_of_week, minute_of_day)
        _, requests, _, _, _,_ = environment.step()
        learning_agent.reset(requests, environment.day_of_week, environment.minute_of_day)
        number_of_steps = int(DURATION / CYCLE - NO_OP_STEPS)

        print("========================================================================================")
        print("EPISODE: {:d} / DATE: {:d} / DAY_OF_WEEK: {:d} / MINUTES: {:.1f} / VEHICLES: {:d}".format(
            episode, date, environment.day_of_week, environment.minute_of_day, number_of_fleets
        ))
        score, _ = run(environment, learning_agent, number_of_steps, average_cycle=AVERAGE_CYCLE, cheat=True)
        describe(score)
        score.to_csv(SCORE_PATH + 'score_dqn' + str(episode) + '.csv')
        print("========================================================================================")
        if episode >= 0 and episode % 2 == 0:
            with open(SCORE_PATH + 'ex_memory_v7' + str(episode) + '.pkl', 'wb') as file:
                pickle.dump(learning_agent.replay_memory, file)


if __name__ == '__main__':
    main()
