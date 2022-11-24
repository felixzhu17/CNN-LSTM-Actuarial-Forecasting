import pickle
import sys
import os
from tqdm import tqdm

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, BASE_PATH)

from methods.config import *
from methods.clean_data import Data_Prep
from methods.build_nn_model import build_model
from methods.nn import get_NN_results
from methods.model_results import get_model_details
from methods.data_methods import prepare_model_data
from methods.shap_methods import NNForecastShap, shap_file_path


END_YEAR = 2019
VARIABLES = ["CPIAUCSL", "GS5", "RPI", "UNRATE"]
OUTPUT_STEPS = 1


def get_shap_object(variable):
    data_prep = Data_Prep(DATA_PATH, TRANSFORM_PATH)
    model_details = get_model_details(END_YEAR, variable, OUTPUT_STEPS)
    look_back_steps = int(model_details["look_back_years"] * 12)
    number_of_pca = model_details["number_of_pca"]

    data_prep.transform_to_supervised_learning(
        NA_CUTOFF,
        [variable],
        OUTPUT_STEPS,
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-01-01",
    )
    dataset = data_prep.supervised_dataset
    full_dataset = dataset["transformed_data"]

    data = prepare_model_data(
        window=full_dataset,
        X_variables=dataset["X_variables"],
        Y_variables=dataset["Y_variables"],
        val_steps=VAL_STEPS,
        look_back=look_back_steps,
        test_steps=TEST_STEPS,
        remove_outlier=REMOVE_OUTLIER,
        number_of_pca=number_of_pca,
        target_variables=dataset["target_variables"],
    )

    data["train_X"] = data["train_X"][len(data["train_X"]) % BATCH_SIZE :]
    data["train_Y"] = data["train_Y"][len(data["train_Y"]) % BATCH_SIZE :]

    model = build_model(data, model_details)

    NN_results = get_NN_results(
        model,
        data,
        VAL_STEPS,
        TEST_STEPS,
        look_back_steps,
        dataset,
        BATCH_SIZE,
        EPOCHS,
        executions=1,
    )
    return NNForecastShap(
        NN_results.test_models[0],
        NN_results.data.test_X,
        VARIABLES_MAP[dataset["target_variables"][0]],
        model_details,
    )


for variable in tqdm(VARIABLES):
    with open(shap_file_path(variable), "wb") as f:
        pickle.dump(get_shap_object(variable), f)
