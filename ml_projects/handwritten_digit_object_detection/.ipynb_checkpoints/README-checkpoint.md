# Object Detection Project

## Project Overview

This project implements an object detection system using three main components:

1. Training code
2. Testing code
3. Visualization code

## System Requirements

- This code must be run on the **NGC-PyTorch-2.3 kernel**.

## Installation

Before running the code, install the following dependencies:

```bash
# PyTorch and torchvision
pip install torch torchvision

# Pillow (PIL)
pip install Pillow

# tqdm
pip install tqdm

# NumPy
pip install numpy

# Torchmetrics
pip install torchmetrics
```

## **Object Detection Project**

## **Project Overview**

This project implements an object detection system using three main components:

1. Training code
2. Testing code
3. Visualization code

## **Usage Instructions**

## **1\. Training Code**

- Generates the weights file for the model
- Pre-trained with optimal parameters
- **Note**: Only run if the provided weights file doesn't work

## **2\. Testing Code**

Input:

- Path to testing images
- Path to testing annotations
- Path to the weights file (from training)

Output:

- MAP (Mean Average Precision)
- Accuracy
- Average IoU (Intersection over Union)

## **3\. Visualization Code**

Input:

- Image path
- Path to the weights file (from training)

Output:

- Predicted bounding boxes on the given image along with digit prediction.

## **Important Notes**

- Ensure consistent weights file naming across all three codes
- If the provided weights file fails:
    1. Run training code with 40 epochs
    2. Estimated time per epoch: 1 minute 44 seconds
    3. Total training time: ~1 hour

## **Submission Details**

- The training code in the submission uses a different weights file name
- This allows verification of the training code without affecting the main weights file
- Ignore the newly created weights file when running the training code for verification

## **Troubleshooting**

If retraining is necessary:

1. Execute the training code
2. Update the weights file name in all three codes to maintain consistency