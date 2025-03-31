# FlightForesight: AI-Powered Airline Delay Prediction

FlightForesight is an AI-driven system that predicts airline delays with high accuracy using deep learning. The system consists of three predictive models:

1. **Binary Classification**: Predicts whether a flight delay will exceed 24 minutes.
2. **Multi-Class Classification**: Categorizes flight delays into four severity levels.
3. **Regression**: Predicts the number of arriving flights at an airport.

---

## Key Features

### 1. Dual Classification & Regression Models
- **Binary Classification**: Detects severe delays (>24 minutes).
- **Multi-Class Classification**: Categorizes flight delays into:
  - **Class 0**: 0–30 minutes
  - **Class 1**: 31–100 minutes
  - **Class 2**: 101–200 minutes
  - **Class 3**: 200+ minutes
- **Regression Model**: Predicts the number of arriving flights based on various factors.

### 2. High-Performance Deep Learning Models
- **Binary Classification Accuracy**: 95.2%
- **Multi-Class Classification Accuracy**: 93.3%
- **Regression Model Performance**:
  - **R² Score**: 0.8891
  - **MAE**: 0.2462
  - **MSE**: 0.1101

### 3. Advanced Feature Engineering & Optimization
- **Handled missing values**:
  - Applied predictive models (**CatBoost, XGBoost, HistGradientBoosting**) for imputation.
  - Selected the best model for each column based on imputation performance.
- **Outlier Detection**- :
  - Used **Isolation Forest** and other techniques to detect extreme values.
  - Standardized features to mitigate the impact of outliers.
- **Skewness Correction & Standardization**:
  - Applied **Yeo-Johnson transformation** to correct skewed distributions.
  - Applied standardization to bring features to a common scale and mitigate the influence of extreme values.
- **Feature Encoding**:
  - **Cyclical Encoding** (sin/cos transformations) for time-based features.
  - **One-Hot Encoding** for categorical variables.
  - **Label Encoding** for ordinal categorical features.
- **Optimized Deep Learning Models**:
  - **Custom activation functions** for better performance.
  - **AdamW optimizer** for faster convergence.
  - **Regularization techniques**:
    - **Dropout layers** to prevent overfitting.
    - **Early stopping** based on validation performance.

### 4. Comprehensive Model Evaluation
#### **Binary Classification**
- **Accuracy**: **95.2%**
- **Confusion Matrix**: Balanced classification performance with strong recall.
- **Precision & Recall**:
  - Class 0: **97% Precision / 94% Recall**
  - Class 1: **93% Precision / 97% Recall**
- **F1-score: **95% for both classes**.

#### **Multi-Class Classification**
- **Accuracy: **93.3%**
- **Confusion Matrix**: High accuracy in major classes.
- **Classification Report**: Strong overall precision, recall, and F1-scores, with variations across classes.

#### **Regression Model**
- **R² Score**: **0.8891**
- **MAE**: **0.2462**
- **MSE**: **0.1101**
- **Median Absolute Error**: **0.1894**

---

---

## Dataset & Preprocessing

- **Data Source**: Publicly available flight delay dataset from Kaggle.
- **Dataset Link**: [Airline Delay Cause Dataset](https://www.kaggle.com/datasets/ramyhafez/airline-delay-cause)
- **Preprocessing Steps**:
  - **Missing Values Handling**: Imputed using predictive modeling.
  - **Outlier Detection & Treatment**: Applied advanced techniques like Isolation Forest.
  - **Feature Engineering**:
    - **Cyclical Encoding** for time-based features (month).
    - **One-Hot & Label Encoding** for categorical variables.
    - **StandardScaler** for numerical features.
    - **Yeo-Johnson Transformation** for skewed features.

---

## Model Architectures

### 1. Binary Classification Model
- **Input**: Feature vector (79 dimensions)
- **Hidden Layers**:
  - Dense (8 neurons, Tanh)
  - Dense (256 neurons, Sigmoid)
  - Dense (128 neurons, Tanh)
  - Dropout (0.3)
  - Dense (64 neurons, Tanh)
  - Dense (32 neurons, Tanh)
  - Dropout (0.2)
- **Output**: 1 neuron (Sigmoid activation)

### 2. Multi-Class Classification Model
- **Input**: Feature vector (79 dimensions)
- **Hidden Layers**:
  - Dense (8 neurons, Tanh)
  - Dense (256 neurons, Sigmoid)
  - Dense (128 neurons, Sigmoid)
  - Dense (64 neurons, Tanh)
  - Dense (32 neurons, Tanh)
  - Dropout (0.2)
- **Output**: 4 neurons (Softmax activation)

### 3. Regression Model
- **Input**: Feature vector (79 dimensions)
- **Hidden Layers**:
  - Dense (8 neurons, Tanh)
  - Dense (256 neurons, Sigmoid)
  - Dense (128 neurons, Tanh)
  - Dropout (0.3)
  - Dense (64 neurons, Tanh)
  - Dense (32 neurons, Tanh)
  - Dropout (0.2)
- **Output**: 1 neuron (Linear activation)

---

## Training Details

### Binary Classification & Multi-Class Classification
- **Optimizer**: AdamW (Binary) / Adam (Multi-class)
- **Loss Functions**:
  - **Binary Classification**: Binary Crossentropy
  - **Multi-Class Classification**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Early Stopping**: Based on validation accuracy

### Regression Model
- **Optimizer**: AdamW
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: MSE
- **Early Stopping**: Based on validation loss

---

## Technology Stack

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy, PowerTransformer (Yeo-Johnson), Feature Scaling & Encoding
- **Model Evaluation**: Confusion Matrix, F1 Score, Precision-Recall Analysis, R² Score, MSE, MAE
- **Optimization**: AdamW Optimizer, Early Stopping, Dropout Regularization
- **Visualization**: Matplotlib, Seaborn, Plotly

---

## Project Highlights

- Developed **three deep learning models** for airline delay prediction.  
- Achieved **high classification accuracy**: **Binary (95.2%)**, **Multi-Class (93.3%)**.  
- Achieved strong regression performance: **R² = 0.8861**.  
- Applied **advanced preprocessing techniques**:
  - Predictive imputation for missing values.
  - Outlier detection and mitigation.
  - Feature transformation & encoding strategies.
- Implemented **regularization strategies**:
  - Dropout, early stopping for better generalization.


---
