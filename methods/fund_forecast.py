import numpy as np
from random import choice
from numpy import quantile
from dateutil.relativedelta import *

from .clean_data import Data_Prep
from .residual_bootstrap import get_residuals

FREQUENCY = "md"
DATA_PATH = f"Data/fred{FREQUENCY}.csv"
TRANSFORM_PATH = f"Data/fred{FREQUENCY}_description.csv"
TARGET_VARIABLES = ["CPIAUCSL", "GS5", "UNRATE", "RPI"]
data_prep = Data_Prep(DATA_PATH, TRANSFORM_PATH)
PERIODS = {
    "During COVID": 2019,
    "After GFC": 2017,
    "During GFC": 2010,
    "Before GFC": 2006,
}


class Pension_Fund:
    def __init__(
        self,
        start_date,
        workers=1500,
        retirees=500,
        benefits=2000,
        contributions=750,
        value=5000,
    ):
        self.retirees = [retirees]
        self.workers = [workers]
        self.benefits = [benefits]
        self.contributions = [contributions]
        self.value = [value]
        self.dates = [start_date]
        self.prev_retirees = retirees
        self.prev_workers = workers
        self.prev_benefits = benefits
        self.prev_contributions = contributions
        self.prev_value = value
        self.prev_date = start_date

    def update_fund(
        self, inflation_rate, investment_returns, wage_rate, unemployment_rate
    ):

        self.prev_retirees = self.prev_retirees
        self.prev_workers = self.prev_workers * (unemployment_rate)
        self.prev_benefits = self.prev_benefits * (inflation_rate)
        self.prev_contributions = self.prev_contributions * (wage_rate)
        self.prev_value = (
            self.prev_value * (investment_returns)
            - self.prev_retirees * self.prev_benefits
            + self.prev_workers * self.prev_contributions
        )
        self.prev_date = self.prev_date + relativedelta(months=+1)
        self.retirees.append(self.prev_retirees)
        self.workers.append(self.prev_workers)
        self.benefits.append(self.prev_benefits)
        self.contributions.append(self.prev_contributions)
        self.value.append(self.prev_value)
        self.dates.append(self.prev_date)


def reverse_diff(first_value, sequence, log_diff=False):
    if log_diff:
        return [i.item() for i in np.exp(np.r_[np.log(first_value), sequence].cumsum())]
    else:
        return [i.item() for i in np.r_[first_value, sequence].cumsum()]


def reverse_transform(results, variables_t, variables_t_1):
    output = results.copy()
    for k, v in output.items():
        if k == "GS5" or k == "UNRATE":
            output[k] = 1 + np.array(reverse_diff(variables_t[k], v)) / 100
        elif k == "RPI":
            output[k] = np.exp(v)
        elif k == "CPIAUCSL":
            output[k] = np.exp(
                reverse_diff(np.log(variables_t[k]) - np.log(variables_t_1[k]), v)
            )
    return output


def forecast_fund_value(
    results, period="After GFC", output_steps=24, interval=False, simulations=5000
):

    point_fund_info, date_t = get_fund_info(results, period, False)
    point_fund_value = calculate_fund_value(point_fund_info, output_steps, date_t)

    actual_fund_info, date_t = get_fund_info(results, period, False, True)
    actual_fund_value = calculate_fund_value(actual_fund_info, output_steps, date_t)

    fund_intervals = None

    if interval:
        paths = []
        for _ in range(simulations):
            fund_info, date_t = get_fund_info(results, period, True)
            fund_value = calculate_fund_value(fund_info, output_steps, date_t)
            paths.append(fund_value["forecast"])

        def get_quantile(q):
            return quantile(paths, q, axis=0)

        fund_intervals = {
            0.4: {"lower": get_quantile(0.15), "upper": get_quantile(0.85)},
            0.1: {"lower": get_quantile(0.05), "upper": get_quantile(0.95)},
            0.05: {"lower": get_quantile(0.025), "upper": get_quantile(0.975)},
            0.01: {"lower": get_quantile(0.995), "upper": get_quantile(0.005)},
        }

    return {
        "point_value": point_fund_value["forecast"],
        "actual_value": actual_fund_value["forecast"],
        "intervals": fund_intervals,
        "dates": point_fund_value["dates"],
    }


def get_fund_info(results, period="After GFC", use_residual=False, actual=False):

    transformed_fund_info = {i: [] for i in TARGET_VARIABLES}

    if period == "During GFC":
        start_index = 1

    else:
        start_index = -1

    for variable in transformed_fund_info.keys():
        target = [
            i
            for i in results
            if variable in i["variables"]
            and i["period"]["end"] == PERIODS[period]
            and i["output_steps"] == 24
        ][0]

        if actual:
            values = target["NN_results"]["test"]["actual_Y"]

        else:
            values = target["NN_results"]["test"]["pred_Y"]

        if use_residual:
            residuals = get_residuals(target["NN_results"])
            transformed_fund_info[variable] = [
                j[start_index] + choice(residuals[i]) for i, j in values.items()
            ]

        else:
            transformed_fund_info[variable] = [
                j[start_index] for i, j in values.items()
            ]

        date_t = target["NN_results"]["dates"]["test"][start_index]
        date_t_1 = target["NN_results"]["dates"]["test"][start_index - 1]

    variables_t = data_prep.raw_data[TARGET_VARIABLES].loc[date_t]
    variables_t_1 = data_prep.raw_data[TARGET_VARIABLES].loc[date_t_1]
    fund_info = reverse_transform(transformed_fund_info, variables_t, variables_t_1)

    return fund_info, date_t


def calculate_fund_value(fund_info, output_steps, date_t):
    NN_fund = Pension_Fund(date_t)
    for i in range(output_steps):
        inflation_rate = fund_info["CPIAUCSL"][i]
        investment_returns = fund_info["GS5"][i]
        wage_rate = fund_info["RPI"][i]
        unemployment_rate = fund_info["UNRATE"][i]
        NN_fund.update_fund(
            inflation_rate, investment_returns, wage_rate, unemployment_rate
        )
    return {"forecast": NN_fund.value, "dates": NN_fund.dates}
