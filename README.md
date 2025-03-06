# FlightForesight: AI-Powered Airline Delay Prediction

FlightForesight is an AI-driven system that predicts airline delays with high accuracy using deep learning. The system consists of three predictive models:

1. *Binary Classification*: Predicts whether a flight delay will exceed 100 minutes.
2. *Multi-Class Classification*: Categorizes flight delays into four severity levels.
3. *Regression*: Predicts the number of arriving flights at an airport.

---

## Key Features

### 1. Dual Classification & Regression Models
- *Binary Classification*: Detects severe delays (>100 minutes).
- *Multi-Class Classification*: Categorizes flight delays into:
  - *Class 0*: 0–30 minutes
  - *Class 1*: 31–100 minutes
  - *Class 2*: 101–200 minutes
  - *Class 3*: 200+ minutes
- *Regression Model*: Predicts the number of arriving flights based on various factors.

### 2. High-Performance Deep Learning Models
- *Binary Classification Accuracy*: 97.2%
- *Multi-Class Classification Accuracy*: 93.62%
- *Regression Model Performance*:
  - *R² Score*: 0.8861
  - *MAE*: 0.2532
  - *MSE*: 0.1149

### 3. Advanced Feature Engineering & Optimization
- *Handled missing values* using imputation techniques.
- *Outlier Detection*: Identified and addressed outliers.
- *Skewness Correction: Applied **PowerTransformer (Yeo-Johnson)* for normalization.
- *Feature Encoding*:
  - *Cyclical Encoding* (sin/cos transformations) for time-based features.
  - *One-Hot Encoding* for categorical variables.
  - *Label Encoding* for ordinal categorical features.
- *Feature Scaling: Applied **StandardScaler* for numerical features.
- *Optimized Deep Learning Models*:
  - *Custom activation functions* for better performance.
  - *AdamW optimizer* for faster convergence.
  - *Regularization techniques*:
    - *Dropout layers* to prevent overfitting.
    - *Early stopping* based on validation performance.

### 4. Comprehensive Model Evaluation
- *Classification Models*: Confusion matrices, precision-recall analysis, F1 scores.
- *Regression Model: Evaluated using **R² Score, MAE, and MSE*.
- *Training Monitoring*: Visualized accuracy and loss curves.

---

## Dataset & Preprocessing

- *Data Source*: Publicly available flight delay dataset from Kaggle.
- *Preprocessing Steps*:
  - *Handled missing values* with imputation techniques.
  - *Identified outliers* and *applied transformations*.
  - *Feature Engineering*:
    - *Cyclical Encoding* for time-based features (e.g., month, day of the week).
    - *One-Hot Encoding* for categorical variables.
    - *Label Encoding* for ordinal categories.
    - *StandardScaler* for numerical features.
    - *Yeo-Johnson Transformation* for skewed features.

---

## Model Architectures

### 1. Binary Classification Model
- *Input*: Feature vector (79 dimensions)
- *Hidden Layers*:
  - Dense (8 neurons, Tanh)
  - Dense (256 neurons, Sigmoid)
  - Dense (128 neurons, Tanh)
  - Dropout (0.3)
  - Dense (64 neurons, Tanh)
  - Dense (32 neurons, Tanh)
  - Dropout (0.2)
- *Output*: 1 neuron (Sigmoid activation)

### 2. Multi-Class Classification Model
- *Input*: Feature vector (79 dimensions)
- *Hidden Layers*:
  - Dense (8 neurons, Tanh)
  - Dense (256 neurons, Sigmoid)
  - Dense (128 neurons, Sigmoid)
  - Dense (64 neurons, Tanh)
  - Dense (32 neurons, Tanh)
  - Dropout (0.2)
- *Output*: 4 neurons (Softmax activation)

### 3. Regression Model
- *Input*: Feature vector (79 dimensions)
- *Hidden Layers*:
  - Dense (8 neurons, Tanh)
  - Dense (256 neurons, Sigmoid)
  - Dense (128 neurons, Tanh)
  - Dropout (0.3)
  - Dense (64 neurons, Tanh)
  - Dense (32 neurons, Tanh)
  - Dropout (0.2)
- *Output*: 1 neuron (Linear activation)

---

## Training Details

### Binary Classification & Multi-Class Classification
- *Optimizer*: AdamW (Binary) / Adam (Multi-class)
- *Loss Functions*:
  - *Binary Classification*: Binary Crossentropy
  - *Multi-Class Classification*: Categorical Crossentropy
- *Metrics*: Accuracy
- *Early Stopping*: Based on validation accuracy

### Regression Model
- *Optimizer*: AdamW
- *Loss Function*: Mean Squared Error (MSE)
- *Metrics*: MSE
- *Early Stopping*: Based on validation loss

---

## Technology Stack

- *Deep Learning*: TensorFlow, Keras
- *Data Processing*: Pandas, NumPy, PowerTransformer (Yeo-Johnson), Feature Scaling & Encoding
- *Model Evaluation*: Confusion Matrix, F1 Score, Precision-Recall Analysis, R² Score, MSE, MAE
- *Optimization*: AdamW Optimizer, Early Stopping, Dropout Regularization
- *Visualization*: Matplotlib, Seaborn

---

## Project Highlights

- Developed three deep learning models for airline delay prediction.  
- Achieved high classification accuracy: *Binary (97.2%), **Multi-Class (93.62%)*.  
- Achieved strong regression performance: *R² = 0.8861*.  
- Applied *outlier detection* and *skewness correction* using *PowerTransformer (Yeo-Johnson)*.  
- Used *advanced feature encoding: **cyclical encoding, one-hot encoding, and label encoding*.  
- Implemented *regularization strategies* (dropout, early stopping) for better generalization.  

---
