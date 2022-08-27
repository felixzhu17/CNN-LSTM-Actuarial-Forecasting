from re import M
import numpy as np
from statsmodels.tsa.api import VAR
from .config import *
from .data_methods import (
    prepare_results,
    split_Y,
    linear_error,
    datetime_to_list,
    BaseResults,
    Dates,
)
from dataclasses import dataclass


@dataclass
class VARResults:
    """Results of VAR model."""

    train: BaseResults
    test: BaseResults
    test_interval: dict
    dates: Dates
    look_back: int
    model: VAR
    val: BaseResults = None
    val_interval: dict = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def get_VAR_results(
    data_info: dict,
    max_look_back: int = 15,
    test_steps: int = 1,
    output_steps: int = 1,
    error_function=linear_error,
    val_steps: int = 0,
):
    data = VAR_array(data_info, test_steps, val_steps)

    # Pred Shapes
    pred_shape = (1, output_steps * len(data_info["target_variables"]))

    if val_steps > 0:
        temp = np.append(data["train"], data["val"], axis=0)
    else:
        temp = data["train"]

    # Select Best Lag
    model = VAR(temp)
    model_fitted = model.fit(maxlags=max_look_back, ic="aic")
    best_lag = model_fitted.k_ar

    # Test Fit
    test_pred_Y = np.empty((0, output_steps * len(data_info["target_variables"])))
    test_interval = {
        alpha: {
            "lower": np.empty((0, output_steps * len(data_info["target_variables"]))),
            "upper": np.empty((0, output_steps * len(data_info["target_variables"]))),
        }
        for alpha in ALPHAS
    }

    for i in range(test_steps):
        fit_data = np.append(temp, data["test"][:i], axis=0)
        forecast = model_fitted.forecast(fit_data, output_steps)
        test_pred_Y = np.append(test_pred_Y, forecast.reshape(pred_shape), axis=0)

        for alpha in ALPHAS:
            forecast, lower, upper = model_fitted.forecast_interval(
                fit_data, output_steps, alpha=alpha
            )
            test_interval[alpha]["lower"] = np.append(
                test_interval[alpha]["lower"], lower.reshape(pred_shape), axis=0
            )
            test_interval[alpha]["upper"] = np.append(
                test_interval[alpha]["upper"], upper.reshape(pred_shape), axis=0
            )

    # Train Fit
    train_model = VAR(data["train"])
    train_model_fitted = train_model.fit(best_lag)
    train_pred_Y = train_model_fitted.fittedvalues

    # Val Fit
    if val_steps > 0:
        val_pred_Y = np.empty((0, output_steps * len(data_info["target_variables"])))
        val_interval = {
            alpha: {
                "lower": np.empty(
                    (0, output_steps * len(data_info["target_variables"]))
                ),
                "upper": np.empty(
                    (0, output_steps * len(data_info["target_variables"]))
                ),
            }
            for alpha in ALPHAS
        }

        for i in range(val_steps):
            fit_data = np.append(data["train"], data["val"][:i], axis=0)
            forecast = train_model_fitted.forecast(fit_data, output_steps)
            val_pred_Y = np.append(val_pred_Y, forecast.reshape(pred_shape), axis=0)

            for alpha in ALPHAS:
                forecast, lower, upper = train_model_fitted.forecast_interval(
                    fit_data, output_steps, alpha=alpha
                )
                val_interval[alpha]["lower"] = np.append(
                    val_interval[alpha]["lower"], lower.reshape(pred_shape), axis=0
                )
                val_interval[alpha]["upper"] = np.append(
                    val_interval[alpha]["upper"], upper.reshape(pred_shape), axis=0
                )

    # Prepare Output
    values = data_info["transformed_data"][data_info["Y_variables"]].values
    actual_Y = split_Y(values, val_steps, test_steps)

    train_results = prepare_results(
        train_pred_Y,
        data["train"][best_lag:],
        error_function,
        data_info["target_variables"],
        data_info["target_variables"],
    )
    test_results = prepare_results(
        test_pred_Y,
        actual_Y["test"],
        error_function,
        data_info["target_variables"],
        data_info["Y_variables"],
    )

    if val_steps > 0:
        val_results = prepare_results(
            val_pred_Y,
            actual_Y["val"],
            error_function,
            data_info["target_variables"],
            data_info["Y_variables"],
        )

        train_dates = data_info["transformed_data"].index[
            best_lag : (len(data_info["transformed_data"]) - test_steps - val_steps)
        ]

        val_dates = data_info["transformed_data"].index[
            (len(data_info["transformed_data"]) - test_steps - val_steps) : (
                len(data_info["transformed_data"]) - test_steps
            )
        ]

        test_dates = data_info["transformed_data"].index[
            (len(data_info["transformed_data"]) - test_steps) :
        ]

        output = VARResults(
            train=train_results,
            test=test_results,
            test_interval=test_interval,
            dates=Dates(
                train=datetime_to_list(train_dates),
                test=datetime_to_list(test_dates),
                val=datetime_to_list(val_dates),
            ),
            look_back=best_lag,
            val=val_results,
            val_interval=val_interval,
            model=train_model_fitted,
        )

    else:
        train_dates = data_info["transformed_data"].index[
            best_lag : (len(data_info["transformed_data"]) - test_steps)
        ]

        test_dates = data_info["transformed_data"].index[
            (len(data_info["transformed_data"]) - test_steps) :
        ]

        output = VARResults(
            train=train_results,
            test=test_results,
            test_interval=test_interval,
            dates=Dates(
                train=datetime_to_list(train_dates), test=datetime_to_list(test_dates)
            ),
            look_back=best_lag,
            model=train_model_fitted,
        )

    return output


def VAR_array(data_info: dict, test_steps: int = 1, val_steps: int = 20):
    """Transform target variables in data_info (window) for training and testing."""

    target_variables = [i for i in data_info["Y_variables"] if i.endswith(f"(t+1)")]
    values = data_info["transformed_data"][target_variables].values
    return split_Y(values, val_steps, test_steps)
