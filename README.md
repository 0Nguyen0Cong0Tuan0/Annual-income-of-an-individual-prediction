# Annual Income Prediction

This project predicts the annual income of individuals based on demographic and employment-related attributes. It involves data preprocessing, feature engineering, and classification using various machine learning models.

## Project Structure
- **`classification.ipynb`**: Implements machine learning models for classification, evaluates their performance, and generates metrics such as specificity, ROC curves, and classification reports.
- **`data_preprocessing.ipynb`**: Handles data cleaning, feature selection, normalization, and discretization to prepare the dataset for modeling.
- **`data/`**: Contains the cleaned dataset (`data_cleaned.csv`) used for training and testing the models.

## Key Features

### Data Preprocessing
- **Data Cleaning**: Handles missing values by replacing placeholders (e.g., `?`) with `NaN` and deciding on appropriate imputation or exclusion strategies.
- **Feature Selection**: Identifies the most informative features using techniques like Mutual Information and correlation analysis.
- **Normalization**: Scales features to a common range to ensure no single feature dominates the analysis.
- **Discretization**: Converts continuous features into categorical bins using techniques like equal-width and equal-depth partitioning.

### Classification
- Implements multiple machine learning models:
  - **Decision Tree**
  - **Random Forest**
  - **Gaussian Naive Bayes**
  - **Bernoulli Naive Bayes**
  - **Multi-Layer Perceptron (MLP)**
- Evaluates models using metrics such as:
  - Confusion Matrix
  - Specificity (True Negative Rate)
  - ROC Curve and AUC
  - Classification Report (Precision, Recall, F1-Score)

## Dataset
The dataset used in this project is the **Adult Census Income Dataset**, which contains demographic and employment-related attributes. The cleaned dataset is saved as `data/data_cleaned.csv`.

### Attributes
- **Features**: Age, workclass, education, marital status, occupation, race, sex, capital gain, capital loss, hours per week, native country, etc.
- **Target**: Binary classification of income (`<=50K` or `>50K`).

## How to Run

1. **Data Preprocessing**:
   - Open and run `data_preprocessing.ipynb` to clean and preprocess the dataset.
   - The cleaned dataset will be saved as `data/data_cleaned.csv`.

2. **Classification**:
   - Open and run `classification.ipynb` to train and evaluate machine learning models.

## Results
The project evaluates the performance of various models and provides metrics such as:
- Specificity (True Negative Rate)
- ROC AUC
- Training Time
- Detailed classification reports

## Dependencies
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `skopt`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Conclusion
This project demonstrates the end-to-end process of data preprocessing and classification for predicting annual income. It highlights the importance of data cleaning, feature engineering, and model evaluation in building robust machine learning pipelines.