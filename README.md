# Project Setup and Usage Guide

## Environment Setup

To set up the required environment, run:

```
pip install -r requirements.txt
```

## Hyperparameter Search

Models were trained by running scripts in the `model_tuning` folder sequentially. You do not need to run this step as it has already been completed. Results are located in the `results` folder and can be analyzed afterwards.

## Training Results

To see a summary of hypertuning results:

1. Run `hypertune_results.ipynb`
2. This notebook:
   - Loads precomputed results from the `results` folder
   - Shows a comparison of results against the benchmark VAR model
   - Displays examples of prediction intervals
   - Presents examples of fund forecasting

This analysis corresponds to Chapter 4 in the paper.

## Model Demo

To train and test the best model structures:

1. Run `model_demo.ipynb`
2. This notebook:
   - Loads the best hyperparameters found from the hyperparameter search
   - Replicates the entire training process end-to-end
   - Trains models across the data
   - Shows test results

## Interpretation

To view PCA components and SHAP feature importance/dependence plots for the associated models:

1. Run `interpretation.ipynb`
2. This notebook generates visualizations corresponding to Appendix D in the paper
