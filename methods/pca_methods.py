from methods.config import *
from methods.clean_data import Data_Prep
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


END_YEAR = 2019
data_prep = Data_Prep(DATA_PATH, TRANSFORM_PATH)
data_prep.transform_to_supervised_learning(
    NA_CUTOFF, [], 1, start=f"{START_YEAR}-01-01", end=f"{END_YEAR}-01-01"
)
dataset = data_prep.supervised_dataset
full_dataset = dataset["transformed_data"]


class PCAExplain:
    def __init__(self, window, X_variables, val_test, test_steps):
        self.window = window
        self.X_variables = X_variables
        self.val_test = val_test
        self.test_steps = test_steps
        self.variable_mapping = {
            f"{k}(t)": v for k, v in DETAILED_VARIABLE_MAPPING.items()
        }
        self.pca = self._get_pca_information(window, X_variables, val_test, test_steps)
        self.pca_component_df = pd.DataFrame(
            self.pca.components_, columns=self.X_variables
        ).T
        self.pca_component_df.index = self.pca_component_df.index.map(
            self.variable_mapping
        )

    def _get_pca_information(self, window, X_variables, val_test, test_steps):
        X_data = window[X_variables]

        pca = PCA()
        scaler = StandardScaler()
        # Fit PCA on Training Only
        fit_pca_data = X_data.iloc[: -(VAL_STEPS + TEST_STEPS)]
        scaled_fit_pca_data = scaler.fit_transform(fit_pca_data)
        pca = pca.fit(scaled_fit_pca_data)
        return pca

    def plot_explained_variance(self):
        fig = px.bar(self.pca.explained_variance_ratio_ * 100)
        fig.update_traces(marker=dict(color="#0052CC"))
        fig.update_layout(
            title=f"PCA Explained Variance",
            xaxis_title="PCA",
            yaxis_title=r"% Explained Variance",
            plot_bgcolor="white",
            showlegend=False,
        )
        return fig

    def plot_pca_component(self, component_number, top_n=10):
        x = self.pca_component_df[component_number].sort_values(
            ascending=False, key=abs
        )[:top_n]
        fig = px.bar(x)
        fig.update_traces(marker=dict(color="#0052CC"))
        fig.update_layout(
            title=f"PCA({component_number}) Top Components",
            xaxis_title="Components",
            yaxis_title=r"Value",
            plot_bgcolor="white",
            showlegend=False,
        )
        return fig


PCAExplain = PCAExplain(full_dataset, dataset["X_variables"], VAL_STEPS, TEST_STEPS)
