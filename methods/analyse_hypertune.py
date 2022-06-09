import os
import pandas as pd
import plotly.express as px
import numpy as np
from .config import *


def hypertune_best_results(results):
    best_results = pd.DataFrame(index=OUTPUT_STEPS, columns=RESULTS_COLUMNS)
    best_results_detailed = []
    for output_steps in OUTPUT_STEPS:
        for period, end_date in PERIODS_MAP.items():
            for target_variable in TARGET_VARIABLES:

                output = [
                    i
                    for i in results
                    if i["output_steps"] == output_steps
                    and i["period"]["end"] == end_date
                    and target_variable in i["variables"]
                ]

                # Sort by validation loss
                output.sort(key=lambda i: i["val_loss"])

                try:
                    best_results.loc[
                        (output_steps), (period, target_variable, "Val")
                    ] = output[0]["val_results"]["average"]
                    best_results.loc[
                        (output_steps), (period, target_variable, "Test")
                    ] = output[0]["test_results"]["average"]
                    best_results_detailed.append(output[0])
                except:
                    pass

    cols = best_results.columns[best_results.dtypes.eq("object")]
    best_results[cols] = best_results[cols].apply(pd.to_numeric, errors="coerce")

    return best_results, best_results_detailed


def aggregate_hypertune_results(best_results):
    aggregate_results = pd.DataFrame(index=OUTPUT_STEPS, columns=AGGREGATE_COLUMNS)
    for output_steps in OUTPUT_STEPS:
        for target_variable in TARGET_VARIABLES:
            aggregate_results.loc[(output_steps), (target_variable, "Val")] = np.mean(
                [
                    best_results.loc[(output_steps), (i, target_variable, "Val")]
                    for i in PERIODS_MAP.keys()
                ]
            )
            aggregate_results.loc[(output_steps), (target_variable, "Test")] = np.mean(
                [
                    best_results.loc[(output_steps), (i, target_variable, "Test")]
                    for i in PERIODS_MAP.keys()
                ]
            )

    cols = aggregate_results.columns[aggregate_results.dtypes.eq("object")]
    aggregate_results[cols] = aggregate_results[cols].apply(
        pd.to_numeric, errors="coerce"
    )

    return aggregate_results


def hypertune_best_fixed_params(results, paramater, parameter_label):
    best_results = pd.DataFrame(
        columns=["Variable", "Output Steps", "Period", parameter_label]
    )
    for output_steps in OUTPUT_STEPS:
        for period, end_date in PERIODS_MAP.items():
            for target_variable in TARGET_VARIABLES:

                output = [
                    i
                    for i in results
                    if i["output_steps"] == output_steps
                    and i["period"]["end"] == end_date
                    and target_variable in i["variables"]
                ]
                output.sort(key=lambda i: i["val_loss"])

                try:
                    best_results.loc[len(best_results)] = [
                        target_variable,
                        output_steps,
                        period,
                        output[0][paramater],
                    ]
                except:
                    best_results.loc[len(best_results)] = [
                        target_variable,
                        output_steps,
                        period,
                        0,
                    ]

    best_results["Variable"] = best_results["Variable"].replace(VARIABLES_MAP)
    best_results["Output Steps"] = best_results["Output Steps"].astype(str)
    best_results[parameter_label] = pd.to_numeric(best_results[parameter_label]).fillna(
        0
    )

    # plot_3d_results(best_results, parameter_label)

    variable_results = best_results.groupby(["Variable"]).mean().transpose()
    period_results = best_results.groupby(["Period"]).mean().transpose()
    best_results["Output Steps"] = best_results["Output Steps"].astype(int)
    steps_results = best_results.groupby(["Output Steps"], sort=True).mean().transpose()

    plot_parameter_results(variable_results, "Variable", parameter_label)
    plot_parameter_results(period_results, "Period", parameter_label)
    plot_parameter_results(steps_results, "Output Steps", parameter_label)

    return variable_results, steps_results, period_results


def hypertune_best_variable_params(results, paramater, parameter_label, ignore_0=False):
    best_results = pd.DataFrame(
        columns=["Variable", "Output Steps", "Period", parameter_label]
    )
    for output_steps in OUTPUT_STEPS:
        for period, end_date in PERIODS_MAP.items():
            for target_variable in TARGET_VARIABLES:

                output = [
                    i
                    for i in results
                    if i["output_steps"] == output_steps
                    and i["period"]["end"] == end_date
                    and target_variable in i["variables"]
                ]
                output.sort(key=lambda i: i["val_loss"])

                try:
                    best_results.loc[len(best_results)] = [
                        target_variable,
                        output_steps,
                        period,
                        output[0]["model_parameters"][paramater],
                    ]

                except:
                    best_results.loc[len(best_results)] = [
                        target_variable,
                        output_steps,
                        period,
                        0,
                    ]

    best_results["Variable"] = best_results["Variable"].replace(VARIABLES_MAP)
    best_results["Output Steps"] = best_results["Output Steps"].astype(str)
    best_results[parameter_label] = pd.to_numeric(best_results[parameter_label]).fillna(
        0
    )

    # plot_3d_results(best_results, parameter_label)

    if ignore_0:
        variable_results = (
            best_results.replace(0, np.NaN).groupby(["Variable"]).mean().transpose()
        )

        period_results = (
            best_results.replace(0, np.NaN).groupby(["Period"]).mean().transpose()
        )
        best_results["Output Steps"] = best_results["Output Steps"].astype(int)
        steps_results = (
            best_results.replace(0, np.NaN)
            .groupby(["Output Steps"], sort=True)
            .mean()
            .transpose()
        )

    else:
        variable_results = best_results.groupby(["Variable"]).mean().transpose()
        period_results = (
            best_results.replace(0, np.NaN).groupby(["Period"]).mean().transpose()
        )
        best_results["Output Steps"] = best_results["Output Steps"].astype(int)
        steps_results = (
            best_results.groupby(["Output Steps"], sort=True).mean().transpose()
        )

    plot_parameter_results(variable_results, "Variable", parameter_label)
    plot_parameter_results(period_results, "Period", parameter_label)
    plot_parameter_results(steps_results, "Output Steps", parameter_label)

    return variable_results, steps_results, period_results


def plot_parameter_results(results, aggregate, parameter):
    x = results.transpose()
    x[aggregate] = x.index.astype(str)
    fig = px.bar(x, x=aggregate, y=parameter, text=parameter, width=1250, height=300)
    fig.update_traces(texttemplate="%{text:.2f}", textfont_size=BIGGEST_SIZE)
    fig.update_layout(
        font=dict(
            size=BIGGEST_SIZE,
        )
    )
    fig.write_image(os.path.join(IMAGE_PATH, f"{aggregate}_{parameter}.png"))
    fig.show()
    return


def plot_3d_results(results, parameter):
    fig = px.scatter_3d(
        results,
        x="Variable",
        y="Output Steps",
        z=parameter,
        color="Variable",
        width=1250,
        height=750,
    )
    fig.update_layout(
        font=dict(
            size=BIGGEST_SIZE,
        )
    )
    fig.show()
    return
