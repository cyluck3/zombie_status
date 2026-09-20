# Zombie Transformation Risk Classifier

An interactive web application built with Streamlit, scikit-learn, Imbalanced-learn, and Plotly to classify individual transformation risk based on physiological, behavioral, and exposure indicators.

## Overview

This project implements a complete machine learning workflow to tackle class imbalance in binary classification tasks. Using a synthetically generated dataset that simulates epidemiological and clinical indicators, the application compares two distinct models:

1. **Logistic Regression**: Serves as a baseline probabilistic classifier with interpretable log-odds coefficients.
2. **Random Forest Classifier**: Implements an ensemble of decision trees to capture non-linear feature interactions.

Both models incorporate **SMOTE (Synthetic Minority Over-sampling Technique)** within an imbalanced-learn pipeline to prevent bias toward the majority class during training.

## Key Features

* **Synthetic Dataset Generation**: Creates 1,500 record benchmark datasets using custom probability scoring derived from physiological parameters (temperature, bites, cognitive coherence, impulsiveness, exposure).
* **Imbalanced Data Handling**: Integrated `imbalanced-learn` pipelines run SMOTE resampling after preprocessing to avoid data leakage during cross-validation.
* **Dual Model Inference**: Real-time side-by-side inference and evaluation metrics comparing Logistic Regression and Random Forest models.
* **Interactive Dashboard**: Configurable sidebar parameter controls that allow users to test edge cases and view dynamic classification probabilities.
* **Data Visualization**: Multi-tab Exploratory Data Analysis (EDA) views using Plotly histograms to analyze key metrics across age and gender distributions.

## Technical Stack

* **Frontend / UI**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Machine Learning & Pipelines**: scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
* **Sampling Methods**: `imbalanced-learn` (`SMOTE`, `ImbPipeline`)
* **Data Visualization**: Plotly Express

## Installation and Local Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/zombie_status.git
   cd zombie-status
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**

   ```bash
   pip install numpy pandas streamlit plotly scikit-learn imbalanced-learn
   ```

4. **Launch the application:**

   ```bash
   streamlit run app.py
   ```

## Machine Learning Architecture

```
Raw Input Data
   │
   ├── Numerical Features ────> Mean Imputer ────> StandardScaler ───┐
   │                                                                ├──> ColumnTransformer
   └── Categorical Features ──> OneHotEncoder ──────────────────────┘
                                                                           │
                                                                           ▼
                                                                  SMOTE Over-sampling
                                                                           │
                                                                           ▼
                                                                 Classifier Evaluation
                                                         (Logistic Regression / Random Forest)
```

### Preprocessing Strategy
* **Numerical Scaling**: Imputed using mean strategy and standard-scaled ($z = \frac{x - \mu}{\sigma}$).
* **Categorical Encoding**: One-hot encoded with unknown category handling.
* **Resampling**: Applied strictly within the training pipeline using SMOTE to balance binary target distribution (`about_to_transform`).

## Repository Structure

```
.
├── app.py              # Main application logic and model definitions
├── requirements.txt    # Environment dependencies
└── README.md           # Project documentation
```
