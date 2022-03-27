import pandas as pd
import numpy as np


class Data_Prep:
    '''Class to store everything related to the data used for testing. Contains the raw datasets, cleaned datasets and transformations for different target variables'''

    def __init__(self, data_path, transform_path):
        # Raw data
        self.raw_data = self.import_FRED_data(data_path)

        # Stock and Watson data description
        self.data_description = pd.read_csv(transform_path)

    def import_FRED_data(self, data_path):
        '''Imports FRED Database from the given data path'''

        import_data = pd.read_csv(data_path)
        data = import_data.drop([0, 1])

        # Date as index
        data['sasdate'] = pd.to_datetime(data['sasdate'])
        data.set_index('sasdate', inplace=True)

        return data

    def remove_NA_data(self, NA_cutoff: int):
        '''Remove NA values of dataset for variables above a certain NA_cutoff'''

        # Determine which variables to remove
        high_na = pd.DataFrame(self.raw_data.isna().sum(), columns=[
                               'NA']).query(f"NA > {NA_cutoff}")
        remove_variables = list(high_na.index)
        remove_variables = [i for i in remove_variables if i != 'WAGE']

        self.na_removed = self.raw_data.drop(
            remove_variables, axis=1, inplace=False)
        self.na_removed = self.na_removed.dropna()

    def transform_FRED_data(self):
        '''Perform Stock and Watson stationarity transformations on cleaned dataset'''

        # Transformation dictionary
        try:
            transformation = dict(
                self.data_description[['fred_mnemonic', 'tcode']].values)
        except:
            transformation = dict(
                self.data_description[['fred', 'tcode']].values)

        temp_data = self.na_removed.copy()

        for i in temp_data.columns:
            tcode = transformation[i]

            if (tcode == 1):
                # Case 1 Level (i.e. no transformation): x(t)
                temp_data[i] = temp_data[i]

            elif (tcode == 2):
                # Case 2 First difference: x(t)-x(t-1)
                temp_data[i] = temp_data[i].diff()

            elif (tcode == 3):
                # case 3 Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
                temp_data[i] = temp_data[i].diff().diff()

            elif (tcode == 4):
                # case 4 Natural log: ln(x)
                temp_data[i] = np.log(temp_data[i])

            elif (tcode == 5):
                # case 5 First difference of natural log: ln(x)-ln(x-1)
                temp_data[i] = np.log(temp_data[i]).diff()

            elif (tcode == 6):
                # case 6 Second difference of natural log:
                # (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
                temp_data[i] = np.log(temp_data[i]).diff().diff()

            elif (tcode == 7):
                # case 7 First difference of percent change:
                # (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)
                temp_data[i] = temp_data[i].pct_change().diff()

        self.transformed_data = temp_data

    def filter_data(self, start=None, end=None):
        self.filtered_data = self.raw_data.loc[start:end]

    def transform_to_supervised_learning(self, NA_cutoff: int, target_variables: list, output_steps: int, start=None, end=None):
        '''
        Transform cleaned dataset to a supervised learning dataset with target variable/s of output_steps. 
        Output is a dataset containing all original variables in period (t), with additional columns for target variables in period (t+i) along with supporting information. 
        This dataset is added to the supervised learning dictionary
         '''
        self.remove_NA_data(NA_cutoff)
        self.transform_FRED_data()

        # Add (t) to column name of each variable
        with_target_transformed = self.transformed_data.copy()
        with_target_transformed.columns = [
            i+"(t)" for i in with_target_transformed.columns]

        # Create period ahead for target variables and add (t+1) to their column names
        for period in range(1, output_steps+1):
            for i in target_variables:
                with_target_transformed[f"{i}(t+{period})"] = with_target_transformed[f"{i}(t)"].shift(
                    periods=-1*period)

        # Remove final observation as there is no future data
        with_target_transformed = with_target_transformed.dropna()

        # Repeat for untransformed
        with_target_raw = self.na_removed.copy()
        with_target_raw.columns = [i+"(t)" for i in with_target_raw.columns]

        # Create period ahead for target variables and add (t+1) to their column names
        for period in range(1, output_steps+1):
            for i in target_variables:
                with_target_raw[f"{i}(t+{period})"] = with_target_raw[f"{i}(t)"].shift(
                    periods=-1*period)

        # Remove final observation as there is no future data
        with_target_raw = with_target_raw.dropna()

        # Name of variables
        X_variables = [
            i for i in with_target_transformed.columns if i.endswith('(t)')]
        Y_variables = [
            i for i in with_target_transformed.columns if not i.endswith('(t)')]

        # Filter Data
        with_target_transformed = with_target_transformed.loc[start:end]
        with_target_raw = with_target_raw.loc[start:end]

        self.supervised_dataset = {"raw_data": with_target_raw, "transformed_data": with_target_transformed, "X_variables": X_variables,
                                   "Y_variables": Y_variables, "description": self.data_description, 'target_variables': target_variables}
