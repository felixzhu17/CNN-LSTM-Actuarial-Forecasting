from tensorflow.keras.optimizers import Adam
#from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def build_model(data, model_details):
    return _build_model(data, model_details)  if "CNN_layers" in model_details['model_parameters'] else _build_model_no_CNN(data, model_details)

def _build_model(data, model_details):
    model = Sequential()
    Learning_rate = model_details['model_parameters']['learning_rate']
    CNN_layers = model_details['model_parameters']['CNN_layers']
    LSTM_layers = model_details['model_parameters']['LSTM_layers']
    Dense_layers = model_details['model_parameters']['Dense_layers']
    Dropout_prob = model_details['model_parameters']['Dropout_prob']

    for i in range(CNN_layers):
        if i == 0:
            model.add(
                Conv1D(filters=model_details['model_parameters'][f'CNN_{i}_filters'],
                    kernel_size=2,
                    input_shape=(
                        data["train_X"].shape[1],
                        data["train_X"].shape[2],
                    ),
                )
            )
        else:
            model.add(
                Conv1D(filters = model_details['model_parameters'][f'CNN_{i}_filters'],
                    kernel_size=2,
                )
            )

        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(Dropout_prob))

    for i in range(LSTM_layers):
        if i == LSTM_layers - 1:
            model.add(
                LSTM(
                    model_details['model_parameters'][f'LSTM_{i}_units'],
                    input_shape=(
                        data["train_X"].shape[1],
                        data["train_X"].shape[2],
                    ),
                )
            )
        else:
            model.add(
                LSTM(
                    model_details['model_parameters'][f'LSTM_{i}_units'],
                    input_shape=(
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
                model_details['model_parameters'][f'Dense_{i}_units'],
                input_shape=(
                        data["train_X"].shape[1],
                        data["train_X"].shape[2],
                    ),
            )
        )
        model.add(Dropout(Dropout_prob))

    model.add(Dense(data['train_Y'].shape[1], activation="linear"))

    opt = Adam(learning_rate=Learning_rate, decay=1e-6)

    # Compile model
    model.compile(loss="mean_squared_error", optimizer=opt)

    return model


def _build_model_no_CNN(data, model_details):
    model = Sequential()

    Learning_rate = model_details['model_parameters']['learning_rate']
    LSTM_layers = model_details['model_parameters']['LSTM_layers']
    Dense_layers = model_details['model_parameters']['Dense_layers']
    Dropout_prob = model_details['model_parameters']['Dropout_prob']


    if LSTM_layers == 0:
        model.add(
            LSTM(
                model_details['model_parameters']['input_LSTM'],
                input_shape=(
                    data["train_X"].shape[1],
                    data["train_X"].shape[2],
                ),
            )
        )
    else:
        model.add(
            LSTM(
                model_details['model_parameters']['input_LSTM'],
                input_shape=(
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
                LSTM(model_details['model_parameters'][f'LSTM_{i}_units'])
            )
        else:
            model.add(
                LSTM(
                    model_details['model_parameters'][f'LSTM_{i}_units'],
                    return_sequences=True,
                )
            )
        model.add(Dropout(Dropout_prob))

    for i in range(Dense_layers):
        model.add(
            Dense(model_details['model_parameters'][f'Dense_{i}_units'])
        )
        model.add(Dropout(Dropout_prob))

    model.add(Dense(data['train_Y'].shape[1], activation="linear"))
    opt = Adam(learning_rate=Learning_rate, decay=1e-6)

    # Compile model
    model.compile(loss="mean_squared_error", optimizer=opt)

    return model
