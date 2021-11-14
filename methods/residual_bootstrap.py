from numpy import quantile
from statistics import stdev


def get_prediction_intervals(NN_results):
    residual_quantiles = get_residual_quantiles(NN_results)

    for alpha in residual_quantiles.keys():
        for variable in residual_quantiles[alpha]['lower'].keys():
            residual_quantiles[alpha]['lower'][variable] = NN_results['test']['pred_Y'][variable] + \
                residual_quantiles[alpha]['lower'][variable]
        for variable in residual_quantiles[alpha]['upper'].keys():
            residual_quantiles[alpha]['upper'][variable] = NN_results['test']['pred_Y'][variable] + \
                residual_quantiles[alpha]['upper'][variable]

    return residual_quantiles


def get_residual_quantiles(NN_results):
    residuals = get_residuals(NN_results)

    def get_quantile(quant):
        return {k: quantile(v, quant) for k, v in residuals.items()}

    residual_quantiles = {0.4: {'lower': get_quantile(0.15), 'upper': get_quantile(0.85)},
                          0.1: {'lower': get_quantile(0.05), 'upper': get_quantile(0.95)},
                          0.05: {'lower': get_quantile(0.025), 'upper': get_quantile(0.975)},
                          0.01: {'lower': get_quantile(0.995), 'upper': get_quantile(0.005)}
                          }

    return residual_quantiles


def get_residuals(NN_results):

    pred = NN_results['val']['pred_Y']
    actual = NN_results['val']['actual_Y']

    if 'val' in NN_results.keys():
        pred = {k: v + NN_results['val']['pred_Y'][k] for k, v in pred.items()}
        actual = {k: v + NN_results['val']['actual_Y'][k]
                  for k, v in actual.items()}

    residuals = {i: list_diff(pred[i], actual[i]) for i in pred.keys()}

    return residuals


def list_diff(list1, list2):
    return [i-j for i, j in zip(list1, list2)]
