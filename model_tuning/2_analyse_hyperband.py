import pickle
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.insert(0, BASE_PATH)

from methods.config import *
from methods.var import get_VAR_results
from methods.nn import get_NN_results, get_model_name
from methods.data_methods import prepare_model_data
from methods.clean_data import Data_Prep
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, Flatten, Dropout
from keras.models import Sequential

def build_model_no_CNN(hp):
    model = Sequential()
    Learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-5])
    LSTM_layers = hp.Int("LSTM_layers", 0, 2, step=1)
    Dense_layers = hp.Int("Dense_layers", 0, 2, step=1)
    Dropout_prob = hp.Float("Dropout_prob", 0.2, 0.5, step=0.3)

    if LSTM_layers == 0:
        model.add(LSTM(hp.Int("input_LSTM", min_value=32, max_value=192, step=32), batch_input_shape=(
            BATCH_SIZE, data['train_X'].shape[1], data['train_X'].shape[2])))
    else:
        model.add(LSTM(hp.Int("input_LSTM", min_value=32, max_value=192, step=32), batch_input_shape=(
            BATCH_SIZE, data['train_X'].shape[1], data['train_X'].shape[2]), return_sequences=True))
    model.add(Dropout(Dropout_prob))

    for i in range(LSTM_layers):
        if i == LSTM_layers - 1:
            model.add(
                LSTM(hp.Int(f"LSTM_{i}_units", min_value=32, max_value=192, step=32)))
        else:
            model.add(LSTM(hp.Int(
                f"LSTM_{i}_units", min_value=32, max_value=192, step=32), return_sequences=True))
        model.add(Dropout(Dropout_prob))

    for i in range(Dense_layers):
        model.add(
            Dense(hp.Int(f"Dense_{i}_units", min_value=32, max_value=192, step=32)))
        model.add(Dropout(Dropout_prob))

    model.add(Dense(len(dataset['Y_variables']), activation='linear'))
    opt = Adam(learning_rate=Learning_rate, decay=1e-6)

    # Compile model
    model.compile(
        loss='mean_squared_error',
        optimizer=opt
    )

    return model


def build_model(hp):
    model = Sequential()
    Learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-5])
    CNN_layers = hp.Int("CNN_layers", 0, 2, step=1)
    LSTM_layers = hp.Int("LSTM_layers", 0, 2, step=1)
    Dense_layers = hp.Int("Dense_layers", 0, 2, step=1)
    Dropout_prob = hp.Float("Dropout_prob", 0.2, 0.5, step=0.3)

    for i in range(CNN_layers):
        if i == 0:
            model.add(Conv1D(filters=hp.Int(f"CNN_{i}_filters", min_value=32, max_value=128, step=32), kernel_size=2,
                             batch_input_shape=(BATCH_SIZE, data['train_X'].shape[1], data['train_X'].shape[2])))
        else:
            model.add(Conv1D(filters=hp.Int(
                f"CNN_{i}_filters", min_value=32, max_value=128, step=32), kernel_size=2))

        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(Dropout_prob))

    for i in range(LSTM_layers):
        if i == LSTM_layers - 1:
            model.add(LSTM(hp.Int(f"LSTM_{i}_units", min_value=32, max_value=192, step=32), batch_input_shape=(
                BATCH_SIZE, data['train_X'].shape[1], data['train_X'].shape[2])))
        else:
            model.add(LSTM(hp.Int(f"LSTM_{i}_units", min_value=32, max_value=192, step=32), batch_input_shape=(
                BATCH_SIZE, data['train_X'].shape[1], data['train_X'].shape[2]), return_sequences=True))

        model.add(Dropout(Dropout_prob))

    if LSTM_layers == 0:
        model.add(Flatten())

    for i in range(Dense_layers):
        model.add(Dense(hp.Int(f"Dense_{i}_units", min_value=32, max_value=192, step=32), batch_input_shape=(
            BATCH_SIZE, data['train_X'].shape[1], data['train_X'].shape[2])))
        model.add(Dropout(Dropout_prob))

    model.add(Dense(len(dataset['Y_variables']), activation='linear'))

    opt = Adam(learning_rate=Learning_rate, decay=1e-6)

    # Compile model
    model.compile(
        loss='mean_squared_error',
        optimizer=opt
    )

    return model


for end_year in PERIODS_MAP.values():
    results = []
    for look_back_year in LOOK_BACK_YEARS:
        for variable in TARGET_VARIABLES:
            for number_of_pca in NUMBER_OF_PCAS:
                for output_steps in OUTPUT_STEPS:

                    look_back_steps = int(look_back_year*12)
                    data_prep = Data_Prep(DATA_PATH, TRANSFORM_PATH)
                    data_prep.transform_to_supervised_learning(NA_CUTOFF, [
                                                               variable], output_steps, start=f'{START_YEAR}-01-01', end=f'{end_year}-01-01')
                    dataset = data_prep.supervised_dataset
                    full_dataset = dataset['transformed_data']
                    NAME = get_model_name(end_year, variable, FREQUENCY, output_steps,
                                          look_back_year, REMOVE_OUTLIER, VAL_YEARS, number_of_pca)

                    tuner = pickle.load(
                        open(os.path.join(TUNING_PATH, f"results/{NAME}.pkl"), "rb"))

                    data = prepare_model_data(window=full_dataset, X_variables=dataset['X_variables'],
                                              Y_variables=dataset['Y_variables'], val_steps=VAL_STEPS, look_back=look_back_steps,
                                              test_steps=TEST_STEPS, remove_outlier=REMOVE_OUTLIER, number_of_pca=number_of_pca,
                                              target_variables=dataset['target_variables'])

                    # Adjust size to match batch
                    data['train_X'] = data['train_X'][len(
                        data['train_X']) % BATCH_SIZE:]
                    data['train_Y'] = data['train_Y'][len(
                        data['train_Y']) % BATCH_SIZE:]

                    best_hp = tuner.get_best_hyperparameters()[0]
                    best_model = tuner.hypermodel.build(best_hp)

                    NN_results = get_NN_results(
                        best_model, data, VAL_STEPS, TEST_STEPS, look_back_steps, dataset,  BATCH_SIZE, EPOCHS, executions=3)

                    data_prep.transform_to_supervised_learning(
                        NA_CUTOFF, TARGET_VARIABLES, output_steps=output_steps, start=f'{START_YEAR}-01-01', end=f'{end_year}-01-01')
                    var_dataset = data_prep.supervised_dataset
                    Var_results = get_VAR_results(
                        var_dataset, test_steps=TEST_STEPS, val_steps=VAL_STEPS, output_steps=output_steps)

                    val_result = {k: v/Var_results['val']['error'][k]
                                  for k, v in NN_results['val']['error'].items()}
                    val_result['average'] = sum(
                        val_result.values())/len(val_result.values())
                    val_result_raw = {k: v/Var_results['val']['error_raw'][k]
                                      for k, v in NN_results['val']['error_raw'].items()}

                    test_result = {k: v/Var_results['test']['error'][k]
                                   for k, v in NN_results['test']['error'].items()}
                    test_result['average'] = sum(
                        test_result.values())/len(test_result.values())
                    test_result_raw = {k: v/Var_results['test']['error_raw'][k]
                                       for k, v in NN_results['test']['error_raw'].items()}

                    info = {'val_loss': NN_results['val_loss'],
                            'val_results': val_result,
                            'val_results_raw': val_result_raw,
                            'test_results': test_result,
                            'test_results_raw': test_result_raw,
                            'model_parameters': best_hp.values,
                            'period': {'start': START_YEAR, 'end': end_year},
                            'look_back_years': look_back_year,
                            'output_steps': output_steps,
                            'Var_results': Var_results,
                            'NN_results': NN_results,
                            'variables': variable,
                            'number_of_pca': number_of_pca
                            }

                    results.append(info)

    with open(os.path.join(RESULTS_PATH, f"{end_year}_results.pkl"), "wb") as f:
        pickle.dump(results, f)
