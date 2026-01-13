# Ship Classification and Detection
Machine learning pipelines for classifying 80×80 satellite patches as ship or no-ship, and detecting ships inside full satellite scenes.

---

1. I coudn't upload data direct to github , due to size issues , I am attching the link to for data and models.You can upload pkl files and data from here.
2. Link : https://drive.google.com/drive/folders/1CwAgM8WFxfa6ubDomqtUaE9M0TOVOezB?usp=drive_link
- If you face any issues with drive, please reach out to me.I have attched same link in canvas as well.
3. Attched the zip file containing models in canvas as well.
3. Data is .zip format , unzip and run the test file.
4. I didn't used any intermediate folder , path is like ships_dataset/ship_data.npy
5. I didn't uploaded the model pkl files here , some files are very huge in size , so I have attched model pkl in drives, without causing confusion.

### Data
| Path | Description |
|------|-------------|
| ships_dataset/ship_data.npy | 4000 RGB image patches (80×80×3) |
| ships_dataset/ship_labels.npy | Labels (0 = no_ship, 1 = ship) |
| ships_dataset/scenes/ | Full satellite scenes for detection |

### Saved Models
| File | Description |
|------|-------------|
| best_Logistic_Regression_noDR.pkl | Logistic Regression (no dimensionality reduction) |
| best_Random_Forest_noDR.pkl | Random Forest (no dimensionality reduction) |
| best_PCA_Logistic_Regression_PCA.pkl | PCA + Logistic Regression |
| best_PCA_Random_Forest_PCA.pkl | PCA + Random Forest |
| best_ISOMAP_LR_manifold.pkl | ISOMAP + Logistic Regression |
| best_ISOMAP_RF_manifold.pkl | ISOMAP + Random Forest |
| best_LLE_LR_manifold.pkl | LLE + Logistic Regression |
| best_LLE_RF_manifold.pkl | LLE + Random Forest |

## Repository Structure
### Notebooks
| File | Description |
|------|-------------|
| training.ipynb | Model training, PCA, manifold learning, hyperparameter tuning, saving models |
| test.ipynb | Load trained models, compute test metrics, and perform ship detection on scenes |
| Project_Report.pdf | All questions and detailed report regarding project |



---

## How the Code Works

1. Load `.npy` dataset files.
2. Flatten each image into a 19,200-dimensional vector.
3. Perform a stratified train-test split.
4. Train three families of models:
   - Baseline models (Logistic Regression, Random Forest)
   - PCA pipelines (dimensionality reduction + classifier)
   - Manifold learning pipelines (ISOMAP and LLE + classifier)
5. Apply cross-validated GridSearch for hyperparameter tuning.
6. Analyze PCA variance, reconstruction, and RMSE.
7. Visualize manifold embeddings in 2D.
8. Save all best-performing models as `.pkl` files.
9. Detect ships in large scenes using non-overlapping 80×80 sliding windows , in testing.

---

## User-Defined Parameters

Below are all user-defined parameters used in the training and testing notebooks.

### 1. Training Notebook Parameters (training.ipynb)

#### Train-Test Split
| Parameter | Meaning | Default |
|----------|----------|---------|
| test_size | Percentage of data used for testing | 0.2 |
| random_state | Reproducibility | 42 |
| stratify | Keep class ratio balanced | labels array (t) |

#### Logistic Regression (No Dimensionality Reduction)
| Parameter | Description | Values |
|----------|-------------|--------|
| clf__C | Regularization strength | [0.001, 0.01, 0.1, 1] |
| clf__penalty | Penalty type | ['l2'] |
| clf__solver | Optimization solver | ['lbfgs'] |
| max_iter | Maximum iterations | 2000 |

#### Random Forest (No Dimensionality Reduction)
| Parameter | Description | Values |
|----------|-------------|--------|
| clf__n_estimators | Number of trees | [50, 100] |
| clf__max_depth | Tree depth | [5, 10] |
| clf__min_samples_split | Minimum samples per split | [5, 10] |
| clf__min_samples_leaf | Minimum leaf size | [5, 10] |
| random_state | Reproducibility | 42 |

---

### PCA Pipelines
| Parameter | Description | Values |
|----------|-------------|--------|
| pca__n_components | Number of PCA components | [50, 100, 120] |
| scaler__with_mean | Centering required for PCA | True |

#### Logistic Regression (with PCA)
Same parameters as LR above.

#### Random Forest (with PCA)
Same RF parameters as above.

---

### Manifold Learning Pipelines

#### ISOMAP Parameters
| Parameter | Description | Values |
|----------|-------------|--------|
| iso__n_components | Embedding dimensions | [10, 20, 50] |
| iso__n_neighbors | Neighbors for graph | [5, 10] |

#### LLE Parameters
| Parameter | Description | Values |
|----------|-------------|--------|
| lle__n_components | Embedding dimensions | [10, 20, 50] |
| lle__n_neighbors | Neighbors for reconstruction | [5, 10] |
| lle__method | LLE method | 'standard' |

#### Classifier parameters inside manifold pipelines:
Logistic Regression: `clf__C`, `clf__penalty`, `clf__solver`  
Random Forest: `clf__n_estimators`, `clf__max_depth`, `clf__min_samples_split`

---

### Cross-Validation and Search Settings
| Parameter | Description | Default |
|----------|-------------|---------|
| cv | Number of folds (Stratified K-Fold) | 3 |
| scoring | Metrics optimized | accuracy, f1_macro |
| refit | Model selected by | f1_macro |
| n_jobs | CPU cores used | -1 |

---

## 2. Testing Notebook Parameters (test.ipynb)

### Inference Time Measurement
| Parameter | Description | Default |
|----------|-------------|---------|
| samples | Number of samples for timing | 200 |

### Scene Detection Function
| Parameter | Description | Default |
|----------|-------------|---------|
| scene_path | Path to the scene image | user-defined |
| model | Loaded model (.pkl) | user-defined |
| window_size | Tile size | 80 |
| stride | Tile movement step | 80 |
| threshold | Probability threshold for ship detection | 0.50 |

Optional internal parameters (in enhanced detection):
| Parameter | Description | Default |
|----------|-------------|---------|
| contrast_boost | Contrast enhancement factor | 1.4–1.6 |
| use_gray | Whether to test grayscale patches | True |

---

## How to Run Training

This notebook will:
- Load the dataset
- Train all baseline, PCA, and manifold models
- Perform hyperparameter tuning
- Save the best-performing models as `.pkl` files

---

## How to Run Testing

Open and run the notebook:

This notebook will:
- Load all trained `.pkl` models
- Compute test accuracy, F1-score, and inference time
- Display confusion matrices
- Run ship detection on full scene images with best model





