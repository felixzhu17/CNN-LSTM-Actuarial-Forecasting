import shap
import pandas as pd
import plotly.express as px
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, ".."))
SHAP_PATH = os.path.abspath(os.path.join(BASE_PATH, "shap_values"))


class NNForecastShap:
    def __init__(self, model, data, target_variable, model_details):
        self.data = data
        self.target_variable = target_variable
        self.data_shape = (
            self.data.shape[0],
            self.data.shape[1] * self.data.shape[2],
        )
        self.model_details = model_details

        self.get_col_names()
        self.get_shap_values(model)
        self.get_data()
        self.get_importance()

    def get_shap_values(self, model):
        explainer = shap.DeepExplainer(model, self.data)
        self.shap_values = explainer.shap_values(self.data)[0].reshape(
            self.data_shape
        )
        self.shap_values = pd.DataFrame(self.shap_values, columns=self.col_names)

    def get_data(self):
        self.data = pd.DataFrame(
            self.data.reshape(self.data_shape), columns=self.col_names
        )

    def get_col_names(self):
        cols = ["PCA_" + str(i) for i in range(self.model_details["number_of_pca"])] + [
            self.target_variable
        ]
        self.time_periods = self.data.shape[1]

        def _get_col_name_time_period(i):
            if i == 0:
                return [f"{col}(t)" for col in cols]
            else:
                return [f"{col}(t-{i})" for col in cols]

        self.col_names = sum(
            [_get_col_name_time_period(i) for i in reversed(range(self.time_periods))],
            [],
        )

    def get_importance(self):
        self.importance = self.shap_values.abs().mean().sort_values(ascending=True)

    def plot_importance(self, n=10, *args, **kwargs):
        fig = px.bar(
            x=self.importance[-n:], y=self.importance.index[-n:], orientation="h"
        )
        fig.update_traces(marker=dict(color="#0052CC"))
        fig.update_layout(
            title=f"{self.target_variable} Importance Plot",
            xaxis_title="Importance",
            yaxis_title="Features",
            plot_bgcolor="white",
            *args, **kwargs
        )
        return fig

    def plot_dependence(self, col, *args, **kwargs):
        fig = px.scatter(x=self.data[col], y=self.shap_values[col])
        fig.update_traces(marker=dict(color="#0052CC"))
        fig.update_layout(
            title=f"{col} SHAP Values",
            xaxis_title=col,
            yaxis_title="SHAP Value",
            plot_bgcolor="white",
            *args, **kwargs
        )
        return fig


def shap_file_path(variable):
    file_name = f"{variable}_shap_values.pkl"
    return os.path.abspath(os.path.join(SHAP_PATH, file_name))
