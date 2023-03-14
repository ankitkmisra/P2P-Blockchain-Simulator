import json
from numpy.random import default_rng

rng, seeds = None, None
with open("utils/seed.json", "r") as f:
    seeds = json.load(f)
    rng = default_rng(seeds["np.random.default_rng"])

# def load_seed(filename="utils/seed.json"):
#     """ loads seed json file. Called by all scripts that need the shared seed value """
#     with open(filename, "r") as f:
#         data = json.load(f)
#         return data