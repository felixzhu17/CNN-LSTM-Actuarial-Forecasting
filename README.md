## Demo.ipynb and Multivariate_Demo.ipynb demonstrate the usage of different models 

- Demo.ipynb - forecasts one economic variable at a time and uses the Consumer Price Index as an example ('CPIAUCSL'.) The models included here are the Dynamic Factor Model (Stock and Watson), Ridge Regression, Lasso Regression, LSTM and CNN.

- Multivariate_Demo.ipynb - forecasts four economic variable at the same time and uses the Consumer Price Index, 3M Treasury Bill, Unemployment Rate and Wage Rate as examples ('CPIAUCSL', 'TB3MS', 'UNRATE', 'AHETPIx'.) The models included here are VAR, LSTM and CNN.

## Economic Methods folder contains helper functions

- clean_data.py - Class to store and clean different datasets with different target variables
- data_methods.py - Functions to split and scale data into training, validation and test sets, ready to be used for modelling.
- model_metrics.py - Functions to evaluate the success of models
- var.py - Functions to test the success of Dimension Reduction techniques
- dimension_reduction.py - Functions to test the success of Dimension Reduction techniques
- neural_networks.py - Functions to test the success of Neural Network techniques

## Data folder contains the relevant datasets.

## Archive folder can be ignored.