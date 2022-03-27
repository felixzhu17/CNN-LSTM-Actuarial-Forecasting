import matplotlib.pyplot as plt
import os
from .residual_bootstrap import get_prediction_intervals
from .config import *

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_variables(dataset, stationary=True):

    names = [i for i in dataset['target_variables']]
    factors = [i+"(t)" for i in dataset['target_variables']]
    number_of_variables = len(dataset['target_variables'])
    fig, axes = plt.subplots(nrows=number_of_variables,
                             ncols=1, dpi=120, figsize=(FIG_SIZE[0], FIG_SIZE[1]*number_of_variables))

    if stationary == True:
        data_name = 'transformed_data'
    else:
        data_name = 'raw_data'

    if number_of_variables == 1:
        data = dataset[data_name][factors[0]]
        axes.plot(data, color='black', linewidth=1)
        plot_decorations(axes, VARIABLES_MAP[names[0]], legend=False)

    else:

        for i, ax in enumerate(axes.flatten()):
            data = dataset[data_name][factors[i]]
            ax.plot(data, color='black', linewidth=1)
            plot_decorations(ax, VARIABLES_MAP[names[i]], legend=False)

    fig.savefig(os.path.join(IMAGE_PATH, 'variable_graph.png'))
    return


def plot_results(result, period, target_variable=None, show_interval=False, alphas=[0.4, 0.1], intervals=None, limit=24, transform_name=True):

    variables = list(result[period]['actual_Y'].keys())
    if target_variable:
        variables = [i for i in variables if target_variable in i]

    variables = variables[:limit]

    number_of_variables = len(variables)

    if number_of_variables == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1, dpi=120, figsize=FIG_SIZE)
        axes.plot(result['dates'][period], result[period]['actual_Y']
                  [variables[0]], label='Actual', color='orange')
        axes.plot(result['dates'][period], result[period]['pred_Y']
                  [variables[0]], label='Predict', color='blue')
        if show_interval:
            for alpha, colour in zip(alphas, COLOURS):
                axes.plot(result['dates'][period], intervals[alpha]['upper']
                          [variables[0]], label=alpha, color=colour, ls='dashed')
                axes.plot(result['dates'][period], intervals[alpha]
                          ['lower'][variables[0]], color=colour, ls='dashed')

        if transform_name:
            try:
                title = VARIABLES_MAP[variables[0][:-5]] + variables[0][-5:]
            except:
                title = VARIABLES_MAP[variables[0][:-6]] + variables[0][-6:]
        else:
            title = variables[0]

        plot_decorations(axes, title)

    else:
        fig, axes = plt.subplots(nrows=min(len(variables), 4), ncols=max(
            int(len(variables)/4), 1), dpi=120, figsize=(20, 10))
        for i, ax in enumerate(axes.flatten()):
            ax.plot(result['dates'][period], result[period]['actual_Y']
                    [variables[i]], label='Actual', color='orange')
            ax.plot(result['dates'][period], result[period]['pred_Y']
                    [variables[i]], label='Predict', color='blue')
            if show_interval:

                for alpha, colour in zip(alphas, COLOURS):
                    ax.plot(result['dates'][period], intervals[alpha]['upper']
                            [variables[i]], label=alpha, color=colour, ls='dashed')
                    ax.plot(result['dates'][period], intervals[alpha]
                            ['lower'][variables[i]], color=colour, ls='dashed')

            if transform_name:
                try:
                    title = VARIABLES_MAP[variables[i]
                                          [:-5]] + variables[i][-5:]
                except:
                    title = VARIABLES_MAP[variables[i]
                                          [:-6]] + variables[i][-6:]
            else:
                title = variables[i]

            plot_decorations(ax, title)
    return


def plot_fund_forecast(fund_forecast, period, alphas=[0.4, 0.1]):
    fig, axes = plt.subplots(nrows=1, ncols=1, dpi=120, figsize=FIG_SIZE)
    axes.plot(fund_forecast['dates'], fund_forecast['point_value'],
              label='Predict', color='blue')
    axes.plot(fund_forecast['dates'], fund_forecast['actual_value'],
              label='Actual', color='orange')

    for alpha, colour in zip(alphas, COLOURS):
        axes.plot(fund_forecast['dates'], fund_forecast['intervals']
                  [alpha]['upper'], label=alpha, color=colour, ls='dashed')
        axes.plot(fund_forecast['dates'], fund_forecast['intervals']
                  [alpha]['lower'], color=colour, ls='dashed')

    plot_decorations(axes, 'Fund Forecast')
    fig.savefig(os.path.join(IMAGE_PATH, f'fund_{period}.png'))
    return


def plot_example_results(best_results_detailed, test_period, alphas=[0.4, 0.1], model='NN'):

    fig, axes = plt.subplots(nrows=4, ncols=1, dpi=120,
                             figsize=(FIG_SIZE[0], FIG_SIZE[1]*4))

    for i, ax in enumerate(axes.flatten()):

        target = [a for a in best_results_detailed if TARGET_VARIABLES[i] in a['variables']
                  and a['period']['end'] == PERIODS_MAP[test_period] and a['output_steps'] == 1][0]

        if model == 'NN':
            result = target['NN_results']
            intervals = get_prediction_intervals(result)

        elif model == 'Var':
            result = target['Var_results']
            intervals = result['test_interval']

        variables = list(result['test']['actual_Y'].keys())
        variables = [a for a in variables if TARGET_VARIABLES[i] in a]

        ax.plot(result['dates']['test'], result['test']['actual_Y']
                [variables[0]], label='Actual', color='orange')
        ax.plot(result['dates']['test'], result['test']['pred_Y']
                [variables[0]], label='Predict', color='blue')

        for alpha, colour in zip(alphas, COLOURS):
            ax.plot(result['dates']['test'], intervals[alpha]['upper']
                    [variables[0]], label=alpha, color=colour, ls='dashed')
            ax.plot(result['dates']['test'], intervals[alpha]
                    ['lower'][variables[0]], color=colour, ls='dashed')

        title = VARIABLES_MAP[variables[0][:-5]] + variables[0][-5:]
        plot_decorations(ax, title)

    fig.savefig(os.path.join(
        IMAGE_PATH, f'forecast_{test_period}_{model}.png'))

    return


def plot_decorations(ax, title, legend=True):
    ax.set_title(title)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=SMALLEST_SIZE)
    if legend:
        ax.legend(loc="upper left")
    ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
    return
