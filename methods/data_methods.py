import numpy as np
from numpy import array, hstack
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None

def prepare_model_data(window, X_variables: list, Y_variables: list,
                       val_steps: int, look_back: int, 
                       test_steps: int = 1,
                       number_of_pca: int = None, remove_outlier=0.005, target_variables: list = None):
    '''
    Split explanatory variables and target variables into training, validation and test set with the optionality of scaling variables.

    Parameters
    ----------
    window: pandas.core.frame.DataFrame
        Dataset containing economic variables of period (t) and (t+i)

    target_variables : list
        List of target variable names of period (t+i)

    output_steps : int
        The periods ahead of target variables (i)

    val_steps: int
        The number of training data to use as the validation set

    look_back : int
        The number of historical periods to include in the explanatory variables (t-i)

    Y_scale: bool
        Whether to perform MinMax scaling on the target variables

    pca: bool
        Whether to apply PCA. If false, MinMaxScaler is used

    pca_data_description: pandas.core.frame.DataFrame
        Stock and Watson data description

    number_of_pca: int
        Number of PCA to include if pca is True

    '''
    window = remove_outliers(window, val_steps, test_steps, remove_outlier)
    X_data = prepare_X(window, X_variables, val_steps, test_steps, number_of_pca, target_variables)
    Y_values = window[Y_variables].values
    output = split_data(X_data, Y_values, look_back,
                        len(Y_variables), val_steps, test_steps)

    return output


def create_principal_components(window, val_steps, test_steps):
    '''Create PCA from dataset (window) based on Stock and Watson (data_description)'''

    pca_data = window

    # Fit PCA on scaled dataset
    pca = PCA()
    scaler = StandardScaler()

    # Fit PCA on Training Only
    fit_pca_data = pca_data.iloc[:-(val_steps+test_steps)]
    scaled_fit_pca_data = scaler.fit_transform(fit_pca_data)
    pca = pca.fit(scaled_fit_pca_data)

    # Apply PCA to full dataset
    factors_reduced = pca.transform(scaler.transform(pca_data))
    pca_dataframe = pd.DataFrame(data=factors_reduced)
    pca_dataframe.index = pca_data.index

    # Rename Columns
    pca_dataframe.columns = [
        "PCA_" + str(i) + "(t)" for i in pca_dataframe.columns]

    return pca_dataframe


def prepare_X(window, X_variables: str, val_steps, test_steps, number_of_pca: int = None, target_variables=None):
    ''' Perform a scaling operation on the dataset (window), either PCA or MinMaxScaler, returning a Numpy Array of variables. '''

    # Extract variables of time (t) only
    X_data = window[X_variables]
    X_scaler = StandardScaler()

    X_pca = create_principal_components(X_data, val_steps, test_steps)

    # Create PCA
    if number_of_pca is not None:
        pca_columns = [
            "PCA_" + str(i) + "(t)" for i in range(number_of_pca)]
        X_pca = X_pca[pca_columns]

    # Merge back target variables

    X_data = X_data[[i + "(t)" for i in target_variables]]
    X_data = pd.concat([X_pca, X_data], axis=1)

    # Scale data again
    fit_scale_data = X_data.iloc[:-(val_steps+test_steps)]
    X_scaler = X_scaler.fit(fit_scale_data)
    X_scaled = X_scaler.transform(X_data)
    return X_scaled


def split_data(X_values, Y_values, look_back, number_of_variables, val_steps: int = 0, test_steps: int = 1):
    ''' Split explanatory and target Numpy arrays into training, validation and test sets '''

    sequences = hstack((X_values, Y_values))

    X, y = list(), list()
    for i in range(len(sequences)):
        # Find the start of the final window
        end_ix = i + look_back
        # Check if the end of the final window exceeds the length of dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-
                                 number_of_variables], sequences[end_ix-1, -number_of_variables:]
        X.append(seq_x)
        y.append(seq_y)

    no_test_X = array(X)[:-test_steps]
    train_X = no_test_X[:-val_steps, :]
    val_X = no_test_X[(len(no_test_X) - val_steps):, :]

    no_test_Y = array(y)[:-test_steps]
    train_Y = no_test_Y[:-val_steps]
    val_Y = no_test_Y[(len(no_test_Y) - val_steps):]

    test_X = array(X)[-test_steps:]
    if test_steps == 1:
        test_X = test_X.reshape((1, train_X.shape[1], train_X.shape[2]))

    test_Y = array(y)[-test_steps:]

    if val_steps == 0:
        return {'train_X': no_test_X, 'train_Y': no_test_Y, 'test_X': test_X, 'test_Y': test_Y}

    else:
        return {'train_X': train_X, 'train_Y': train_Y, 'val_X': val_X, 'val_Y': val_Y, 'test_X': test_X, 'test_Y': test_Y}

def remove_outliers(data, val_steps, test_steps, remove_outlier=0.005):
    train = data.iloc[:-(val_steps + test_steps)]
    val_test = data.iloc[-(val_steps + test_steps):]

    # Remove outlier
    lower_quantile = train.quantile(remove_outlier)
    upper_quantile = train.quantile(1 - remove_outlier)
    outliers_lower = (train < lower_quantile)
    outliers_upper = (train > upper_quantile)
    train.mask(outliers_lower, lower_quantile, axis=1, inplace=True)
    train.mask(outliers_upper, upper_quantile, axis=1, inplace=True)
    values = pd.concat([train, val_test])
    return values


def split_Y(Y_values, val_steps, test_steps):

    test_Y = Y_values[-test_steps:]
    no_test_Y = Y_values[:-test_steps]
    train_Y = no_test_Y[:-val_steps]
    val_Y = no_test_Y[(len(no_test_Y) - val_steps):]

    if val_steps == 0:
        return {'train': no_test_Y, 'test': test_Y}

    else:
        return {'train': train_Y, 'val': val_Y, 'test': test_Y}


def prepare_results(pred, actual, error_function, target_variables, Y_variables):


    pred_results = {i: [j[count] for j in pred]
                    for count, i in enumerate(Y_variables)}
    # except:
    #     pred_results = {i: [j[0][count] for j in pred]
    #                     for count, i in enumerate(Y_variables)}
    actual_values = {i: [j[count] for j in actual]
                     for count, i in enumerate(Y_variables)}

    error = error_function(pred, actual)
    error_results = {i: np.mean([j[count] for j in error])
                     for count, i in enumerate(Y_variables)}

    # Convert from numpy to native
    # try:
    #     error_results = {k: v.item() for k, v in error_results.items()}
    #     pred_results = {k: [i.item() for i in v]
    #                     for k, v in pred_results.items()}
    #     actual_values = {k: [i.item() for i in v]
    #                      for k, v in actual_values.items()}
    # except:
    #     pass

    output = {'error_raw': error_results,
              'pred_Y': pred_results, 'actual_Y': actual_values}

    # Aggregate error
    error_dict = {i: [] for i in target_variables}

    for i in target_variables:
        for k, v in output['error_raw'].items():
            if k.startswith(i):
                error_dict[i].append(v)

    aggregate_error = {k: sum(v)/len(v) for k, v in error_dict.items()}
    output['error'] = aggregate_error

    return output

def linear_error(pred_y, actual_y):
    return (pred_y - actual_y)**2



