import pickle
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, BASE_PATH)

from methods.analyse_hypertune import hypertune_best_results
from methods.config import *

years = [2006, 2010, 2017, 2019]
results = []
for year in years:
    results.extend(
        pickle.load(open(os.path.join(RESULTS_PATH, f"{year}_results.pkl"), "rb"))
    )

best_results, best_results_detailed = hypertune_best_results(results)
with open(MODEL_INFO, "wb") as f:
    pickle.dump(best_results_detailed, f)
