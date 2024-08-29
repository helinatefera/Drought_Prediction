#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:59:39 2024

@author: Helina Tefera, Mulugeta Berhe, Wendirad Demelash
"""
import abc
import functools
import pathlib
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (LSTM, Bidirectional, Conv1D, ConvLSTM2D, Dense,
                          Dropout, Flatten, Input, MaxPooling1D, RepeatVector,
                          Reshape, TimeDistributed)
from keras.models import Sequential
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

random_seed = 27

random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

dataset_dir = "timeseries_normal.csv"

df = pd.read_csv(dataset_dir, index_col="DATE", parse_dates=True)


class DroughtPredictor:
    """
    A class to train and evaluate drought prediction models.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: list,
        groupby: str,
        n_test_sets: int,
        n_steps_in: int,
        n_steps_out: int,
        architecture: str,
        n_row: int = None,
        features: list = None,
        train_size: float = 0.9,
        normalizer: BaseEstimator = None,
        date_column: str = "DATE",
        latitude_column: str = "GEOGR2",
    ) -> None:
        """Initialize the class with the given parameters."""

        self.df: pd.DataFrame = df
        self.target: list = target
        self.groupby: str = groupby
        self.n_test_sets: int = n_test_sets
        self.n_steps_in: int = n_steps_in
        self.n_steps_out: int = n_steps_out
        self.architecture: str = architecture
        self.n_row: int = n_row
        self.train_size: float = train_size
        self.features: list = features
        # Use the entire colunm if not provided
        if self.features is None:
            self.features = self.df.columns.to_list()

        self.normalizer: BaseEstimator = normalizer
        self.date_column: str = date_column
        self.latitude_column: str = latitude_column

    @functools.cache
    def get_predictor(self) -> "DroughtPredictor":
        """Return an instance of the architecture class."""
        params = {
            attr: value
            for attr, value in self.__dict__.items()
            if not attr.startswith("_")  # Exclude private attributes
        }
        return self._architecture_cls(**params)

    @functools.cached_property
    def _architecture_cls(self) -> "DroughtPredictor":
        """Return the architecture class."""
        return self._get_architectures[self.architecture]

    @functools.cached_property
    def _get_architectures(self) -> dict:
        """Return a dictionary of all subclasses of the class."""
        subclasses = self.__class__.__subclasses__()
        return {cls.name: cls for cls in subclasses}

    @property
    def n_features(self) -> int:
        """Return number of feature variables."""
        return len(self.features)

    @property
    def n_target(self) -> int:
        """Return number of targe variables."""
        return len(self.target)

    @property
    def groups(self) -> pd.core.groupby.DataFrameGroupBy:
        """Return grouped datapoints."""
        return self.df.groupby(self.groupby)

    @property
    def n_col(self):
        return int(self.n_steps_in / self.n_row)

    def has_normalizer(self) -> bool:
        """Return true if a normalizer has given to the class, false otherwise"""
        return self.normalizer is not None

    def get_data(self) -> tuple[list, list]:
        """Return train , test data and PET data for the model."""

        train_datas = []  # Train and validation data
        test_datas = []  # Test datapoints for evaluation
        pet_datas = []  # PET data for each group

        for _, group in self.groups:
            # breakpoint()
            group_data = group[self.features].sort_values(self.date_column).values

            # Normalize the data if normalizer has provided
            if self.has_normalizer():
                group_data = self.normalizer.fit_transform(group_data)

            latitude = group[self.latitude_column][0]
            start_year = group.index[-self.n_test_sets].year
            calibration_year_initial = group.index.min().year
            calibration_year_final = group.index.max().year

            pet_datas.append(
                (latitude, start_year, calibration_year_initial, calibration_year_final)
            )

            seq_X, seq_y = self.to_sequnce(group_data)  # (X, y)
            train_datas.append((seq_X[: -self.n_test_sets], seq_y[: -self.n_test_sets]))
            test_datas.append((seq_X[-self.n_test_sets :], seq_y[-self.n_test_sets :]))

        return train_datas, test_datas, pet_datas

    def to_sequnce(self, sequence: np.array) -> tuple[np.array, np.array]:
        """Generate multivariate multi-step input output data."""
        n = len(sequence)
        X, y = [], []

        for i in range(n):
            inp_end_idx = i + self.n_steps_in
            out_end_idx = inp_end_idx + self.n_steps_out

            # check if we are beyond the dataset
            if out_end_idx > n:
                break

            seq_x = sequence[i:inp_end_idx, :]
            seq_y = sequence[inp_end_idx:out_end_idx, :]

            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)

    def to_test_val(
        self, data_stack: np.array
    ) -> list[list[np.array, np.array], list[np.array, np.array]]:
        """Split the given data in Train and Validation dataset"""
        split_stack = []
        for X, y in data_stack:
            X_train, X_val = self.split_train_val(X)
            y_train, y_val = self.split_train_val(y)

            split_stack.append(((X_train, y_train), (X_val, y_val)))
        return split_stack

    def split_train_val(self, data: np.array) -> tuple[np.array, np.array]:
        n = int(len(data) * self.train_size)
        return data[:n], data[n:]

    def reshape(self, data):
        """Reshape data if it necessary."""
        return data  # No reshape by default

    def train_model(
        self,
        train_stack: list,
        test_stack: list,
        pet_datas: list,
        predictor: Sequential,
        epochs: int = 1000,
        batch_size: int = 128,
        no_plot: bool = False,
        no_history_plot: bool = False,
        no_prediction_plot: bool = False,
    ) -> list:
        test_predictions = []
        for i, data in enumerate(train_stack):
            ((X_train, y_train), (X_val, y_val)) = data

            print(
                f"Training model {i+1} out of {len(train_stack)} split\n"
                f"X_train shape: {X_train.shape}\n"
                f"y_train shape: {y_train.shape}\n"
                f"X_val shape: {X_val.shape}\n"
                f"y_val shape: {y_val.shape}\n"
            )

            X_train = predictor.reshape(X_train)
            X_val = predictor.reshape(X_val)

            model = predictor.create_model()

            model.compile(optimizer="adam", loss="mse", metrics=["mse"])
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=0.001,
                patience=10,
                mode="min",
                restore_best_weights=True,
            )
            reduce_lro = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                mode="min",
                patience=3,
                min_lr=0.001,
            )
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lro],
                verbose=2,
            )

            X_test, y_test = test_stack[i]
            X_test = predictor.reshape(X_test)

            evaluation = model.evaluate(X_test, y_test, verbose=2)
            print(f"Split {i+1}: Evaluation Results - {evaluation}")

            if not (no_plot or no_history_plot):
                self.plot_training_history(history.history, ["loss", "val_loss"])

            pred_y = self.predict_on_test_split(model, X_test)
            if self.has_normalizer():
                y_test = self.normalizer.inverse_transform(
                    np.reshape(y_test, (-1, self.n_features))
                )
                pred_y = self.normalizer.inverse_transform(
                    np.reshape(pred_y, (-1, self.n_features))
                )

            test_predictions.append((y_test, pred_y, *pet_datas[i]))
            if not (no_plot or no_prediction_plot):
                self.plot_predictions(y_test, pred_y)

        return test_predictions

    def predict_on_test_split(self, model, X_test: np.array) -> list:
        test_predictions = []

        for test in X_test:
            test_predictions.append(
                model.predict(test.reshape(-1, *test.shape), verbose=0)
            )
        return np.array(test_predictions)

    def plot_training_history(self, history, metrics=["loss"]):
        results = [(metric, history[metric]) for metric in metrics]

        # Plotting training and validation loss
        plt.figure(figsize=(8, 4))
        for metric, result in results:
            plt.plot(result, label=metric)
        plt.title("Training Across Epochs")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def plot_predictions(self, true_y, pred_y):
        plt.figure(figsize=(8, 5))
        for feature_index in range(self.n_features):
            plt.plot(
                true_y[:, feature_index], label=f"Actual {self.features[feature_index]}"
            )
            plt.plot(
                pred_y[:, feature_index],
                label=f"Predicted {self.features[feature_index]}",
            )

        plt.legend(loc="best")
        plt.show()

    def execute(self, **training_params) -> None:
        """Run the model training pipeline."""
        train_stack, test_stack, pet_datas = self.get_data()
        split_train_stack = self.to_test_val(train_stack)

        predictor = self.get_predictor()

        return self.train_model(
            split_train_stack, test_stack, pet_datas, predictor, **training_params
        )

    @abc.abstractmethod
    def create_model(self) -> Sequential:
        raise NotImplementedError()

    def dump_reult(self, result, path=".", *args) -> bool:
        filename = (
            path
            + "/"
            + "_".join(
                [
                    self.architecture,
                    f"spei{self.n_steps_out}",
                    f"inp{self.n_steps_in}",
                    f"testset{self.n_test_sets}",
                    *args,
                    ".pkl",
                ]
            )
        )
        file = open(filename, "wb")
        pickle.dump(result, file)
        return True


class VanillaLSTM(DroughtPredictor):
    name = "vanilla_lstm"

    def create_model(self) -> Sequential:
        # Define the model
        model = Sequential()

        # Add Input layer
        model.add(Input(shape=(self.n_steps_in, self.n_features)))

        # Add LSTM layer
        model.add(LSTM(100, activation="relu", return_sequences=True))

        # Add Flatten layer
        model.add(Flatten())

        # Add Dense layers with Dropout
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))

        # Add output Dense layer and Reshape
        model.add(Dense(self.n_steps_out * self.n_features))
        model.add(Reshape((self.n_steps_out, self.n_features)))

        return model


class StackedLSTM(DroughtPredictor):
    name = "stacked_lstm"

    def create_model(self) -> Sequential:
        # Define the model
        model = Sequential()

        # Add Input layer
        model.add(Input(shape=(self.n_steps_in, self.n_features)))

        # model.add(LSTM(150, activation="relu", return_sequences=True))
        model.add(LSTM(100, activation="relu", return_sequences=True))
        model.add(LSTM(50, activation="relu"))

        # Add Flatten layer
        model.add(Flatten())

        # Add Dense layers with Dropout
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))

        # Add output Dense layer and Reshape
        model.add(Dense(self.n_steps_out * self.n_features))
        model.add(Reshape((self.n_steps_out, self.n_features)))

        return model


class BiDirectionalLSTM(DroughtPredictor):
    name = "bidirection_lstm"

    def create_model(self) -> Sequential:
        # Define the model
        model = Sequential()

        # Add Input layer
        model.add(Input(shape=(self.n_steps_in, self.n_features)))

        model.add(Bidirectional(LSTM(100, activation="relu")))

        # Add Flatten layer
        model.add(Flatten())

        # Add Dense layers with Dropout
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))

        # Add output Dense layer and Reshape
        model.add(Dense(self.n_steps_out * self.n_features))
        model.add(Reshape((self.n_steps_out, self.n_features)))

        return model


class StackedBiDirectionalLSTM(DroughtPredictor):
    name = "stacked_bidirection_lstm"

    def create_model(self) -> Sequential:
        # Define the model
        model = Sequential()

        # Add Input layer
        model.add(Input(shape=(self.n_steps_in, self.n_features)))

        # model.add(Bidirectional(LSTM(150, activation="relu", return_sequences=True)))
        model.add(Bidirectional(LSTM(100, activation="relu", return_sequences=True)))
        model.add(Bidirectional(LSTM(50, activation="relu")))

        # model.add(LSTM(50, activation='relu'))

        # Add Flatten layer
        model.add(Flatten())

        # Add Dense layers with Dropout
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))

        # Add output Dense layer and Reshape
        model.add(Dense(self.n_steps_out * self.n_features))
        model.add(Reshape((self.n_steps_out, self.n_features)))

        return model


class CNNLSTM(DroughtPredictor):
    name = "cnn_lstm"

    def reshape(self, data):
        return data.reshape((data.shape[0], self.n_row, self.n_col, self.n_features))

    def create_model(self) -> Sequential:
        # Define the model
        model = Sequential()

        # Add Input layer
        model.add(Input(shape=(None, self.n_col, self.n_features)))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation="relu")))
        model.add(
            TimeDistributed(
                Conv1D(filters=64, kernel_size=1, activation="relu"),
            )
        )
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(50, activation="relu"))

        # Add Flatten layer
        model.add(Flatten())

        # Add Dense layers with Dropout
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))

        # Add output Dense layer and Reshape
        model.add(Dense(self.n_steps_out * self.n_features))
        model.add(Reshape((self.n_steps_out, self.n_features)))

        return model


class ConvLSTM(DroughtPredictor):
    name = "conv_lstm"

    def reshape(self, data):
        return data.reshape((data.shape[0], self.n_row, 1, self.n_col, self.n_features))

    def create_model(self) -> Sequential:
        # Define the model
        model = Sequential()

        # Add Input layer
        model.add(Input(shape=(self.n_row, 1, self.n_col, self.n_features)))
        model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu'))

        # Add Flatten layer
        model.add(Flatten())

        # Add Dense layers with Dropout
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))

        # Add output Dense layer and Reshape
        model.add(Dense(self.n_steps_out * self.n_features))
        model.add(Reshape((self.n_steps_out, self.n_features)))

        return model



columns = ["PERCIPT", "TMPMAX", "TMPMIN"]

scaler = MinMaxScaler(feature_range=(0, 1))

next_pred_month = 3

dp = DroughtPredictor(
    df,
    architecture="conv_lstm",
    target=columns,
    groupby="GH_ID",
    n_test_sets=10,
    n_steps_in=24,
    n_steps_out=next_pred_month,
    n_row=6,
    features=columns,
    normalizer=scaler,
)

result = dp.execute(batch_size=32, no_history_plot=True)

result_path = pathlib.Path(f"./results/{next_pred_month }").as_posix()
dp.dump_reult(result, result_path, "")
