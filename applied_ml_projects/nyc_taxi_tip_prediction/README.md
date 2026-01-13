
# Project

This repository contains a project for predicting **tip amount** and **fare amount** for NYC Yellow Taxi rides using **Linear Regression** and **Llinear  Regression with lasso** models.  

## Dataset - NYC Yellow Taxi Dataset

The dataset you will be working with has been pushed to this repository under the name "yellow_taxi_data.csv".

## Repository Files

| File Name | Description |
|-----------|-------------|
| `training.ipynb` | Notebook used for training models, hyperparameter tuning, and saving trained pipelines (`.pkl` files). |
| `test.ipynb` | Notebook for evaluating the final trained models on a test dataset. Uses pre-trained pipeline objects only; no retraining. |
| `yellow_taxi_data.csv` | Dataset used for training and testing. |
| `tip_linear_model.pkl` | Pre-trained Linear Regression pipeline for tip prediction. |
| `tip_linear_lasso_model.pkl` | Pre-trained Lasso Regression pipeline for tip prediction. |
| `fare_linear_model.pkl` | Pre-trained Linear Regression pipeline for fare prediction. |
| `fare_linear_lasso_model.pkl` | Pre-trained Lasso Regression pipeline for fare prediction. |
| `Project_Report.pdf` | This file, contains the detailed report of how we implemented the project. |
| `README.md` | This file, containing instructions and descriptions. |
| `Project 1.ipynb` | This file, containing requirements for project 1 |

## How to Use

### 1. Training
- Open `training.ipynb`.
- Clean the dataset if required (removing rows with `passenger_count == 0` or negative numeric values).
- Run all cells to:
  - Preprocess features using pipelines (`Date_time_total_encoder`, scaling, encoding).
  - Train Linear and Lasso regression models.
  - Save trained pipelines as `.pkl` files (`joblib.dump`).

**User-defined parameters you may adjust in `training.ipynb`:**  
- Test size for train-validation split (`test_size` in `train_test_split`)  
- Random seed (`random_state`)  
- Lasso regularization parameter (`alpha`)  
- Features included in the model (modify 'tip_robust_numeric_features','tip_categorical_features','fare_robust_numeric_features','fare_categorical_features') 
- Names of the .pkl files

### 2. Testing

- Open test.ipynb.
- Load the pre-trained pipelines using joblib.load().
- Prepare the test dataset using the same preprocessing pipeline as used during training.
- Run predictions on test data for both tip and fare models.
- Evaluate model performance using metrics like R², RMSE, or MAE.
- visualize prediction results vs actual values.

**User-defined parameters you may adjust in `testing.ipynb`:**  
- Test size for train-validation split (`test_size` in `train_test_split`)  
- Random seed , select the same random seed used in training.

### Notes:

- No model retraining is performed in test.ipynb.
- Make sure the .pkl files are present in the same directory as test.ipynb.
- Make use yellow_taxi_data.csv and all .pkl files are present
- I used python 3 kernal and jupyter notebook
- if any installation is missing , please install according to the requirements.