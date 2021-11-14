import pickle
import os
import sys
import tensorflow as tf

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.insert(0, BASE_PATH)

from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners import Hyperband
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from methods.nn import get_model_name
from methods.data_methods import prepare_model_data
from methods.clean_data import Data_Prep
from methods.config import *

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

                    x = os.listdir(os.path.join(TUNING_PATH, "results"))
                    if f"{NAME}.pkl" in x:
                        print(f'{NAME} exists')

                    else:
                        data = prepare_model_data(window=full_dataset, X_variables=dataset['X_variables'], Y_variables=dataset['Y_variables'], val_steps=VAL_STEPS, look_back=look_back_steps,
                                                  test_steps=TEST_STEPS, remove_outlier=REMOVE_OUTLIER, number_of_pca=number_of_pca,
                                                  target_variables=dataset['target_variables'])

                        # Adjust size to match batch
                        data['train_X'] = data['train_X'][len(
                            data['train_X']) % BATCH_SIZE:]
                        data['train_Y'] = data['train_Y'][len(
                            data['train_Y']) % BATCH_SIZE:]

                        if look_back_year < 1:
                            NN_func = build_model_no_CNN
                        else:
                            NN_func = build_model

                        tuner = Hyperband(
                            NN_func,
                            objective='val_loss',
                            max_epochs=15,
                            executions_per_trial=5,
                            hyperband_iterations=3,
                            directory="Tuning",
                            overwrite=False,
                            project_name=f"{NAME}"
                        )

                        tuner.search(x=data['train_X'],
                                     y=data['train_Y'],
                                     verbose=2,
                                     epochs=15,
                                     batch_size=BATCH_SIZE,
                                     callbacks=[tf.keras.callbacks.EarlyStopping(
                                         "val_loss", patience=3)],
                                     validation_data=(data['val_X'], data['val_Y']))

                        with open(os.path.join(TUNING_PATH, f"results/{NAME}.pkl"), "wb") as f:
                            pickle.dump(tuner, f)
