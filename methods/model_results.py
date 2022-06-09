import pickle
from .config import *

best_models = pickle.load(open(MODEL_INFO, "rb"))


def get_best_model_name(end_year, target_variable, output_steps):
    return f"1960-{end_year}-{target_variable}-1x{output_steps}.h5"


def get_model_details(end_year, target_variable, output_steps):
    return [
        i
        for i in best_models
        if i["period"]["end"] == end_year
        and target_variable in i["variables"]
        and i["output_steps"] == output_steps
    ][0]
