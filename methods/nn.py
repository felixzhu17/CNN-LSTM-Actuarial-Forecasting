import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
class PredictorData:
    train: np.array
    test: np.array

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclass
class NNResults:
    train: BaseResults
    test: BaseResults
    dates: Dates
    test_models: list
    data: PredictorData
    val: BaseResults = None
    val_loss: float = None
    val_models: list = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def get_NN_results(
    model,
    data,
    val_steps,
    test_steps,
    look_back,
    data_info,
    batch_size,
    epochs,
    executions=5,
    error_function=linear_error,
    data_name="transformed_data",
):
    """Fits the Neural Network on training set and calculates error on training, validation and test set. Best hyperparameters are sorted by validation error in .analyse_hypertune"""

    if val_steps == 0:
        train_X = data["train_X"]
        train_Y = data["train_Y"]
    else:
        train_X = np.concatenate((data["train_X"], data["val_X"]), axis=0)
        train_Y = np.concatenate((data["train_Y"], data["val_Y"]), axis=0)

    full_X = np.concatenate((train_X, data["test_X"]), axis=0)
    full_Y = np.concatenate((train_Y, data["test_Y"]), axis=0)

    predictions = []
    test_model_list = []

    for i in range(executions):

        # Retrain on training and validation
        print(f"Fitting: {i+1}")

        temp_model = duplicate_model(model)
        _ = temp_model.fit(
            x=train_X,
            y=train_Y,
            verbose=0,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping("loss", patience=5)],
        )
        predictions.append(temp_model.predict(full_X, batch_size=batch_size))
        test_model_list.append(temp_model)

    full_pred_Y = np.mean(predictions, axis=0)

    pred_Y = split_Y(full_pred_Y, val_steps, test_steps)
    actual_Y = split_Y(full_Y, val_steps, test_steps)

    data_length = len(data_info[data_name])

    train_results = prepare_results(
        pred_Y["train"],
        actual_Y["train"],
        error_function,
        data_info["target_variables"],
        data_info["Y_variables"],
    )
    train_dates = data_info[data_name].index[
        look_back - 1 : (data_length - test_steps - val_steps)
    ]
    train_dates = train_dates[-len(data["train_X"]) :]
    test_dates = data_info[data_name].index[(data_length - test_steps) :]

    test_results = prepare_results(
        pred_Y["test"],
        actual_Y["test"],
        error_function,
        data_info["target_variables"],
        data_info["Y_variables"],
    )

    if val_steps > 0:

        val_model_list = []
        val_loss = []

        for _ in range(executions):
            temp_model = duplicate_model(model)
            val_history = temp_model.fit(
                x=data["train_X"],
                y=data["train_Y"],
                verbose=0,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(data["val_X"], data["val_Y"]),
                callbacks=[tf.keras.callbacks.EarlyStopping("loss", patience=5)],
            )
            val_loss.append(val_history.history["val_loss"][-1])
            val_model_list.append(temp_model)

        val_loss = np.mean(val_loss)
        val_results = prepare_results(
            pred_Y["val"],
            actual_Y["val"],
            error_function,
            data_info["target_variables"],
            data_info["Y_variables"],
        )
        val_dates = data_info[data_name].index[
            (len(data_info[data_name]) - test_steps - val_steps) : (
                data_length - test_steps
            )
        ]

        output = NNResults(
            train=train_results,
            test=test_results,
            dates=Dates(
                train=datetime_to_list(train_dates),
                test=datetime_to_list(test_dates),
                val=datetime_to_list(val_dates),
            ),
            test_models=test_model_list,
            data=PredictorData(train=train_X, test=full_X),
            val=val_results,
            val_loss=val_loss,
            val_models=val_model_list,
        )

    else:

        output = NNResults(
            train=train_results,
            test=test_results,
            dates=Dates(
                train=datetime_to_list(train_dates), test=datetime_to_list(test_dates)
            ),
            test_models=test_model_list,
            data=PredictorData(train=train_X, test=full_X),
        )
    return output


def get_model_name(
    end_year,
    target_variable,
    frequency,
    output_steps,
    look_back_years,
    remove_outlier,
    val_years,
    number_of_pca,
):
    return f"1960-{end_year}-{target_variable}-{frequency}-{output_steps}-AHEAD-{look_back_years}-LOOKBACK-{remove_outlier}-OUTLIER-{val_years}-VAL-{number_of_pca}-PCA"


def visualize_loss(history, title):
    loss = history.history["loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    try:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "r", label="Validation loss")
    except:
        pass
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def duplicate_model(model):
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=model.optimizer, loss=model.loss)
    return new_model
