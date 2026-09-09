# Credit Card Fraud Detection

## ## Overview

This project presents an end-to-end analysis and machine learning workflow for credit card fraud detection.

The project starts with a detailed **data analysis and exploratory data analysis (EDA)** using Python, Pandas, NumPy, Matplotlib and Seaborn. The objective is to understand the structure and quality of the dataset, analyze transaction characteristics, identify patterns in fraudulent transactions, study feature distributions and correlations, and highlight the challenges associated with the data.

Based on the insights obtained during the analysis, the project then moves to the **machine learning stage**. Several approaches are tested to handle the severe class imbalance between legitimate and fraudulent transactions, followed by the training and comparison of different classification models.

The machine learning workflow includes **Logistic Regression, Random Forest and XGBoost**, with model evaluation based primarily on Precision, Recall, F1-Score, ROC-AUC and PR-AUC rather than accuracy alone.

The project also includes **classification threshold optimization** to find a better trade-off between fraud detection and false positives. The final XGBoost model is saved and integrated into a **Streamlit application**, allowing users to obtain fraud predictions from transaction data.

Overall, the project follows a complete workflow:

**Data Analysis → Exploratory Data Analysis → Data Preparation → Machine Learning → Model Evaluation → Threshold Optimization → Prediction**

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

The dataset contains 284,807 transactions and 31 columns.

The main columns are:

- `Time`: time elapsed since the first transaction in the dataset
- `V1` to `V28`: anonymized numerical features obtained from a PCA transformation
- `Amount`: transaction amount
- `Class`: target variable

The target variable is binary:

- `0`: legitimate transaction
- `1`: fraudulent transaction

The `V1` to `V28` features are anonymized, so the project focuses on their statistical and predictive behavior rather than assigning a specific real-world meaning to individual variables.

## Project Objectives

The main objectives of the project are:

1. Explore and understand the dataset.
2. Clean the data while preserving the original dataset.
3. Analyze the characteristics of fraudulent transactions.
4. Establish a Logistic Regression baseline.
5. Test different strategies for handling class imbalance.
6. Compare several machine learning models.
7. Optimize the classification threshold.
8. Save the final model for later use.
9. Build a simple Streamlit application for interactive predictions.

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_imbalance_strategies.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_threshold_optimization.ipynb
│
├── src/
│   ├── predict.py
│   └── testpredict.py
│
├── models/
│   └── fraud_model.pkl
│
├── app.py
├── README.md
└── .gitignore
```

## Data Exploration

The first part of the project focuses on understanding the dataset.

The initial analysis checks:

- Dataset dimensions
- Data types
- Missing values
- Duplicate transactions
- Class distribution
- Transaction amounts
- Feature distributions
- Time distribution
- Correlations between variables

The dataset contains a strong class imbalance, which is one of the main challenges of the project.

Duplicate rows were removed during the cleaning stage. The original dataset is kept unchanged in `data/raw/`, while the cleaned dataset is stored in `data/processed/`.

## Exploratory Data Analysis

The EDA focuses on identifying patterns that may help distinguish fraudulent transactions from legitimate ones.

Several anonymized features show noticeable differences between the two classes. In particular, features such as `V14`, `V17`, `V12`, `V10` and `V16` show relatively strong correlations with the target variable.

These correlations are used as exploratory observations and are not interpreted as causal relationships.

The analysis also examines transaction amounts and the distribution of transactions over time.

## Baseline Model

Logistic Regression is used as the baseline model.

The purpose of the baseline is to establish a reference point before testing more complex models.

The baseline model achieved the following results on the test set:

Precision : 0.85
Recall : 0.58
F1-Score : 0.69
ROC-AUC : 0.956
PR-AUC : 0.692

The results show that the baseline model can identify fraudulent transactions reasonably well, but its recall is relatively low.

Because the dataset is highly imbalanced, PR-AUC, precision, recall and F1-score are particularly important for this project.

## Handling Class Imbalance

Several approaches were tested to address the class imbalance.

### Class Weighting

Logistic Regression was trained using `class_weight="balanced"`.

This increased fraud recall significantly, but it also produced a large number of false positives. As a result, precision and F1-score decreased.

### Random Oversampling

Random Oversampling was used to increase the number of fraudulent examples in the training data.

The minority class was duplicated until both classes had the same number of samples.

The approach improved recall but resulted in a substantial decrease in precision.

### SMOTE

SMOTE was also tested to generate synthetic minority-class examples.

The results were similar to those obtained with random oversampling. Although fraud recall increased, the number of false positives remained high.

Overall, the imbalance strategies demonstrated an important trade-off between detecting more fraudulent transactions and avoiding false alarms.

## Model Comparison

Three main models were compared:

- Logistic Regression
- Random Forest
- XGBoost

The same train/test split was used for the different models to make the comparison consistent.

| Model               | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| ------------------- | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression |      0.85 |   0.58 |     0.69 |   0.956 |  0.692 |
| Random Forest       |      0.97 |   0.73 |     0.83 |   0.924 |  0.788 |
| XGBoost             |      0.96 |   0.72 |     0.82 |   0.978 |  0.813 |

Random Forest achieved the highest F1-score at the default threshold of 0.5.

XGBoost achieved the best ROC-AUC and PR-AUC among the tested models, making it the most promising model for further optimization.

## Threshold Optimization

The default classification threshold of 0.5 is not necessarily the best choice for fraud detection.

Instead of automatically classifying a transaction as fraudulent when its predicted probability is greater than 0.5, several thresholds were tested.

Thresholds between 0.05 and 0.95 were evaluated using precision, recall and F1-score.

The threshold of `0.10` produced the highest F1-score among the tested values.

Using this threshold with XGBoost produced the following results:

| Metric          | Result |
| --------------- | -----: |
| Precision       |   0.96 |
| Recall          |   0.75 |
| F1-Score        |   0.84 |
| False Positives |      3 |
| False Negatives |     24 |
| True Positives  |     71 |
| True Negatives  | 56,648 |

The lower threshold increases the number of fraudulent transactions detected while keeping the number of false positives very low on the test set.

## Final Model

XGBoost was selected as the final model.

The model was saved together with the optimized classification threshold in:

```text
models/fraud_model.pkl
```

The saved file contains:

- The trained XGBoost model
- The optimized decision threshold

This makes it possible to reuse the trained model without retraining it every time a prediction is required.

## Prediction Module

The prediction logic is implemented in:

```text
src/predict.py
```

The module loads the saved model and threshold and provides a `predict_fraud()` function.

The function receives the transaction features and returns:

- The predicted fraud probability
- The final classification

The feature order is explicitly controlled to ensure that the input format matches the data used during model training.

## Streamlit Application

A Streamlit application was created to make the model easier to demonstrate.

The application is available through:

```text
app.py
```

It allows users to enter:

- Transaction time
- Transaction amount
- The 28 anonymized features

After submitting the transaction, the application displays:

- Fraud probability
- Final prediction
- The classification threshold used by the model

The application uses the saved XGBoost model, so predictions do not require retraining the model.

## Running the Project

### 1. Install the dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

From the project root:

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## Evaluation Strategy

The project does not rely on accuracy as the main evaluation metric because of the strong class imbalance.

The main metrics used are:

**Precision**

Measures how many transactions classified as fraudulent are actually fraudulent.

**Recall**

Measures how many of the actual fraudulent transactions are detected by the model.

**F1-Score**

Provides a balance between precision and recall.

**ROC-AUC**

Measures the model's ability to distinguish between the two classes across different classification thresholds.

**PR-AUC**

Provides a more informative view of performance when the positive class is highly imbalanced.

## Limitations

This project is based on a public dataset containing anonymized features. The model therefore cannot provide an interpretation of what each individual `V` feature represents in real-world transaction terms.

The threshold was optimized on a fixed test set and according to F1-score. In a real fraud detection system, the threshold would normally depend on the operational cost of false positives and false negatives.

The dataset is also historical, so performance on new transactions may differ from the reported test results.

## Future Improvements

Possible improvements include:

- Hyperparameter tuning for XGBoost
- Cross-validation
- More detailed threshold analysis based on business costs
- Model explainability using SHAP
- Automated testing
- API deployment
- Dockerization
- Cloud deployment
- Monitoring model performance over time

## Conclusion

This project demonstrates a complete machine learning workflow for credit card fraud detection.

The experiments show that handling class imbalance is important, but simply increasing the weight or number of fraudulent examples does not necessarily produce the best overall model.

XGBoost provided the strongest overall ranking performance, and threshold optimization improved its F1-score while maintaining high precision.

The final model is saved and integrated into a Streamlit application, making the project suitable both for experimentation and for demonstrating how a trained fraud detection model can be used in practice.
