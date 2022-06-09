import pickle
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, BASE_PATH)

from methods.config import *
from methods.model_results import get_best_model_name, best_models
from methods.nn import get_model_name
from methods.data_methods import prepare_model_data
from methods.clean_data import Data_Prep
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, Flatten, Dropout
from keras.models import Sequential


def build_model_no_CNN(hp):
    model = Sequential()
    Learning_rate = hp.Choice("learning_rate", values=[1e-3, 1e-5])
    LSTM_layers = hp.Int("LSTM_layers", 0, 2, step=1)
    Dense_layers = hp.Int("Dense_layers", 0, 2, step=1)
    Dropout_prob = hp.Float("Dropout_prob", 0.2, 0.5, step=0.3)

    if LSTM_layers == 0:
        model.add(
            LSTM(
                hp.Int("input_LSTM", min_value=32, max_value=192, step=32),
                batch_input_shape=(
                    BATCH_SIZE,
                    data["train_X"].shape[1],
                    data["train_X"].shape[2],
                ),
            )
        )
    else:
        model.add(
            LSTM(
                hp.Int("input_LSTM", min_value=32, max_value=192, step=32),
                batch_input_shape=(
                    BATCH_SIZE,
                    data["train_X"].shape[1],
                    data["train_X"].shape[2],
                ),
                return_sequences=True,
            )
        )
    model.add(Dropout(Dropout_prob))

    for i in range(LSTM_layers):
        if i == LSTM_layers - 1:
            model.add(
                LSTM(hp.Int(f"LSTM_{i}_units", min_value=32, max_value=192, step=32))
            )
        else:
            model.add(
                LSTM(
                    hp.Int(f"LSTM_{i}_units", min_value=32, max_value=192, step=32),
                    return_sequences=True,
                )
            )
        model.add(Dropout(Dropout_prob))

    for i in range(Dense_layers):
        model.add(
            Dense(hp.Int(f"Dense_{i}_units", min_value=32, max_value=192, step=32))
        )
        model.add(Dropout(Dropout_prob))

    model.add(Dense(len(dataset["Y_variables"]), activation="linear"))
    opt = Adam(learning_rate=Learning_rate, decay=1e-6)

    # Compile model
    model.compile(loss="mean_squared_error", optimizer=opt)

    return model


def build_model(hp):
    model = Sequential()
    Learning_rate = hp.Choice("learning_rate", values=[1e-3, 1e-5])
    CNN_layers = hp.Int("CNN_layers", 0, 2, step=1)
    LSTM_layers = hp.Int("LSTM_layers", 0, 2, step=1)
    Dense_layers = hp.Int("Dense_layers", 0, 2, step=1)
    Dropout_prob = hp.Float("Dropout_prob", 0.2, 0.5, step=0.3)

    for i in range(CNN_layers):
        if i == 0:
            model.add(
                Conv1D(
                    filters=hp.Int(
                        f"CNN_{i}_filters", min_value=32, max_value=128, step=32
                    ),
                    kernel_size=2,
                    batch_input_shape=(
                        BATCH_SIZE,
                        data["train_X"].shape[1],
                        data["train_X"].shape[2],
                    ),
                )
            )
        else:
            model.add(
                Conv1D(
                    filters=hp.Int(
                        f"CNN_{i}_filters", min_value=32, max_value=128, step=32
                    ),
                    kernel_size=2,
                )
            )

        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(Dropout_prob))

    for i in range(LSTM_layers):
        if i == LSTM_layers - 1:
            model.add(
                LSTM(
                    hp.Int(f"LSTM_{i}_units", min_value=32, max_value=192, step=32),
                    batch_input_shape=(
                        BATCH_SIZE,
                        data["train_X"].shape[1],
                        data["train_X"].shape[2],
                    ),
                )
            )
        else:
            model.add(
                LSTM(
                    hp.Int(f"LSTM_{i}_units", min_value=32, max_value=192, step=32),
                    batch_input_shape=(
                        BATCH_SIZE,
                        data["train_X"].shape[1],
                        data["train_X"].shape[2],
                    ),
                    return_sequences=True,
                )
            )

        model.add(Dropout(Dropout_prob))

    if LSTM_layers == 0:
        model.add(Flatten())

    for i in range(Dense_layers):
        model.add(
            Dense(
                hp.Int(f"Dense_{i}_units", min_value=32, max_value=192, step=32),
                batch_input_shape=(
                    BATCH_SIZE,
                    data["train_X"].shape[1],
                    data["train_X"].shape[2],
                ),
            )
        )
        model.add(Dropout(Dropout_prob))

    model.add(Dense(len(dataset["Y_variables"]), activation="linear"))

    opt = Adam(learning_rate=Learning_rate, decay=1e-6)

    # Compile model
    model.compile(loss="mean_squared_error", optimizer=opt)

    return model


for i in best_models:
    period = i["period"]
    variable = i["variables"]
    output_steps = i["output_steps"]
    end_year = period["end"]
    look_back_year = i["look_back_years"]
    number_of_pca = i["number_of_pca"]
    look_back_steps = int(LOOK_BACK_YEARS * 12)

    NAME = get_model_name(
        end_year,
        variable,
        FREQUENCY,
        output_steps,
        look_back_year,
        REMOVE_OUTLIER,
        VAL_YEARS,
        number_of_pca,
    )
    tuner = pickle.load(open(os.path.join(TUNING_PATH, f"results/{NAME}.pkl"), "rb"))

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

    best_hp = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hp)

    best_model_name = get_best_model_name(end_year, variable, output_steps)
    best_model.save(os.path.join(MODELS_PATH, best_model_name))
