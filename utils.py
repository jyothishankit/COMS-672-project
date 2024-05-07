from constants import *
from hyperparameters import *

def load_graph():
    with open(GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)
    return G

def load_eta_model():
    with open(ETA_MODEL_PATH, 'rb') as f:
        eta_model = pickle.load(f)
    return eta_model