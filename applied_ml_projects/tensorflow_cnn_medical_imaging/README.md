# Project 3 – Artificial Neural Networks (ChestMNIST & RetinaMNIST)

This repository contains all code, trained models, learning-curve histories, and evaluation notebooks for Project 3 of Artificial Neural Networks. The goal is to train and evaluate Convolutional Neural Networks (CNNs) on two medical imaging datasets: **ChestMNIST (multi-label lung X-rays)** and **RetinaMNIST (multi-class retina images)**.
### I didn't uploaded the datset , please use the dataset avialbale in canvas for project .
### Using path : Project 3/chestmnist.npz and  Project 3/retinamnist_128.npz to load the data
### I have used Tensorflow-2.16 in hipergator for running this project.
### If any issues is there with model keras files or hisrory pkl files , please let me know. 
## Directory Structure
```
project-3-grad-sonamjose
├── train_chest.ipynb
├── train_retina.ipynb
├── test.ipynb
├── model_chest.keras
├── model_retina.keras
├── history_chest.pkl
├── history_retina.pkl
└──Project_Report.pdf
Project 3/
      ├── chestmnist.npz
      └── retinamnist_128.npz
```

## Installation
pip install tensorflow keras numpy matplotlib scikit-learn seaborn

##  User-Defined Parameters

| Parameter | Default Value | Description |
|----------|---------------|-------------|
| `batch_size` | 256 (Chest), 32 (Retina) | Training batch size |
| `learning_rate` | 1e-4 |Initial learning rate for Adam |
| `patience` | 5 | Early stopping threshold |
| `reduce_lr_patience` | 3 | Reduce LR on plateau |
| `min_lr` | 1e-6 | Minimum LR allowed |
| `epochs` | 30 (Chest), 30 (Retina) | Max training epochs |
| `img_size` | (28,28,1) or (128,128,3) | Input size per dataset |

These can be modified in the **training.ipynb** notebook.

##  File Descriptions

### **1. training.ipynb**
This notebook trains both neural network models:
- Loads and preprocesses ChestMNIST and RetinaMNIST datasets.
- Builds the CNN architectures.
- Compiles the models with appropriate loss functions and metrics.
- Trains using best practices:
  - ModelCheckpoint  
  - EarlyStopping  
  - ReduceLROnPlateau  
- Saves:
  - `model_chest.keras`
  - `model_retina.keras`
  - `history_chest.pkl`
  - `history_retina.pkl`

---

### **2. test.ipynb**
This notebook evaluates the trained models:
- Loads the saved `.keras` models.
- Loads test datasets.
- Computes:
  - Test loss  
  - Test accuracy  
  - Test AUC  
  - Weighted F1 and Weighted Accuracy  
- Generates confusion matrices.
- Reloads and plots learning curves from saved history files.
- Produces performance summaries for report.

---

### **3. model_chest.keras**
Saved best-performing **ChestMNIST CNN model**.
- Selected using validation binary accuracy.
- Used only during testing and inference.

---

### **4. model_retina.keras**
Saved best-performing **RetinaMNIST CNN model**.
- Selected using validation categorical accuracy.
- Used during testing for classification.

---

### **5. history_chest.pkl**
Pickle file containing the full training history for ChestMNIST:
- loss  
- binary accuracy  
- AUC  
- validation loss  
- validation accuracy  
- learning rate changes  

Used to replot learning curves in the test notebook and report.

---

### **6. history_retina.pkl**
Pickle file containing the full training history for RetinaMNIST:
- loss  
- accuracy  
- AUC  
- validation loss  
- validation accuracy  
- learning rate changes  

Also used for learning curve visualization.




## 1. How to Run the Training Code

### Step 1 - Open the training notebook

    training.ipynb

### Step 2 - Confirm dataset files exist

    chestmnist.npz
    retinamnist_128.npz

### Step 3 --- Run all cells

Running the notebook will:

-   Load and preprocess both datasets\
-   Build CNN models for ChestMNIST and RetinaMNIST\
-   Train the models\
-   Save trained models:
    -   model_chest.keras\
    -   model_retina.keras\
-   Save training histories:
    -   history_chest.pkl\
    -   history_retina.pkl

------------------------------------------------------------------------

##  How to Run the Test Code

### Step 1 --Open:

    test.ipynb

### Step 2 --- Run all cells

This notebook will:

-   Load trained models\
-   Load the test datasets\
-   Evaluate performance (loss, accuracy, AUC, F1-score)\
-   Generate confusion matrices\
-   Reload & display learning curves from `.pkl` files

------------------------------------------------------------------------

##  4. Using the Code With a Different Test Set

The provided models can be applied to any new dataset that follows the same input format as ChestMNIST or RetinaMNIST.  
To use a custom test set, users must ensure that:

1. **Image dimensions match the expected input shape**  
   - ChestMNIST: `(28, 28, 1)` grayscale  
   - RetinaMNIST: `(128, 128, 3)` RGB  

2. **Pixel values are scaled to the range `[0, 1]`**, just as during training.

3. **Multi-label vs. multi-class format is preserved**  
   - ChestMNIST requires a **binary vector of length 14** for labels.  
   - RetinaMNIST requires a **single integer label (0–4)** or a **one-hot vector**.

4. **No dataset-specific preprocessing is required**  
   As long as the new dataset matches the shape and normalization steps, the trained `.keras` models can be used directly for inference.

5. **Users should adjust directory paths** when loading new images and labels.

### New ChestMNIST-style data:

``` python
X_new = new_images.astype("float32") / 255.0
X_new = X_new.reshape(-1, 28, 28, 1)
pred = model_chest.predict(X_new)
```

### New RetinaMNIST-style data:

``` python
X_new = new_images.astype("float32") / 255.0
pred = model_retina.predict(X_new)
pred_labels = pred.argmax(axis=1)
```

#### Predict probabilities & convert to binary labels

``` python
y_prob = model_chest.predict(X_new)
y_pred = (y_prob >= 0.5).astype(int)
```

####  Compute weighted accuracy & F1-score

``` python
from sklearn.metrics import accuracy_score, f1_score

acc = accuracy_score(t_new.flatten(), y_pred.flatten())
f1 = f1_score(t_new.flatten(), y_pred.flatten(), average='weighted')

print("Weighted Accuracy:", acc)
print("Weighted F1 Score:", f1)
```

#### Confusion matrix for one disease label (example: label 0)

``` python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(t_new[:, 0], y_pred[:, 0])
print(cm)
```

This example shows how to evaluate the model when **your new dataset is
not part of the original .npz files**.



##  5. Output Files Generated

-   model_chest.keras\
-   model_retina.keras\
-   history_chest.pkl\
-   history_retina.pkl\
-   Confusion matrices\
-   Classification reports\
-   Learning curve plots