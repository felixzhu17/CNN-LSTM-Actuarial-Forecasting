import pickle
import os
import sys
import json

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, BASE_PATH)

from methods.config import *
from methods.var import get_VAR_results
from methods.nn import get_NN_results
from methods.data_methods import prepare_model_data
from methods.clean_data import Data_Prep
from methods.model_results import get_model_details
from methods.build_nn_model import build_model


results = []
for end_year in PERIODS_MAP.values():
    for variable in TARGET_VARIABLES:
            for output_steps in OUTPUT_STEPS:

                
                data_prep = Data_Prep(DATA_PATH, TRANSFORM_PATH)
                data_prep.transform_to_supervised_learning(
                    NA_CUTOFF,
                    [variable],
                    output_steps,
                    start=f"{START_YEAR}-01-01",
                    end=f"{end_year}-01-01",
                )
                dataset = data_prep.supervised_dataset
                full_dataset = dataset["transformed_data"]
 
                model_details = get_model_details(end_year, variable, output_steps)
                look_back_steps = int(model_details["look_back_years"] * 12)
                number_of_pca = model_details["number_of_pca"]
                data = prepare_model_data(
                    window=full_dataset,
                    X_variables=dataset["X_variables"],
                    Y_variables=dataset["Y_variables"],
                    val_steps=VAL_STEPS,
                    look_back=look_back_steps,
                    test_steps=TEST_STEPS,
                    remove_outlier=REMOVE_OUTLIER,
                    number_of_pca=number_of_pca,
                    target_variables=TARGET_VARIABLES,
                )

                # Adjust size to match batch
                data["train_X"] = data["train_X"][
                    len(data["train_X"]) % BATCH_SIZE :
                ]
                data["train_Y"] = data["train_Y"][
                    len(data["train_Y"]) % BATCH_SIZE :
                ]

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
                    executions=3,
                )

                data_prep.transform_to_supervised_learning(
                    NA_CUTOFF,
                    TARGET_VARIABLES,
                    output_steps=output_steps,
                    start=f"{START_YEAR}-01-01",
                    end=f"{end_year}-01-01",
                )
                var_dataset = data_prep.supervised_dataset
                Var_results = get_VAR_results(
                    var_dataset,
                    test_steps=TEST_STEPS,
                    val_steps=VAL_STEPS,
                    output_steps=output_steps,
                )

                val_result = {
                    k: v / Var_results["val"]["error"][k]
                    for k, v in NN_results["val"]["error"].items()
                }
                val_result["average"] = sum(val_result.values()) / len(
                    val_result.values()
                )
                val_result_raw = {
                    k: v / Var_results["val"]["error_raw"][k]
                    for k, v in NN_results["val"]["error_raw"].items()
                }

                test_result = {
                    k: v / Var_results["test"]["error"][k]
                    for k, v in NN_results["test"]["error"].items()
                }
                test_result["average"] = sum(test_result.values()) / len(
                    test_result.values()
                )
                test_result_raw = {
                    k: v / Var_results["test"]["error_raw"][k]
                    for k, v in NN_results["test"]["error_raw"].items()
                }

                info = {
                    "val_loss": NN_results["val_loss"],
                    "val_results": val_result,
                    "val_results_raw": val_result_raw,
                    "test_results": test_result,
                    "test_results_raw": test_result_raw,
                    "model_parameters": model_details["model_parameters"],
                    "period": {"start": START_YEAR, "end": end_year},
                    "look_back_years": model_details["look_back_years"],
                    "output_steps": output_steps,
                    "Var_results": Var_results,
                    "NN_results": NN_results,
                    "variables": variable,
                    "number_of_pca": number_of_pca,
                }

                results.append(info)

with open(os.path.join(RESULTS_PATH, f"new_results.json"), "wb") as f:
    json.dump(results, f)
