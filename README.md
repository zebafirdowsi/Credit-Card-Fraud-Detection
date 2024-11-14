
# Credit Card Fraud Detection

## Project Overview

This project focuses on building a machine learning model to detect fraudulent credit card transactions. Using different classification algorithms, we aim to identify potentially fraudulent transactions from a dataset of credit card transactions. The models are trained and evaluated on various metrics like accuracy, sensitivity, specificity, F1-score, and ROC-AUC.

## Key Features

- **Data Preprocessing**: Handle missing data, feature scaling, and feature engineering.
- **Class Imbalance Handling**: Address class imbalance using techniques like SMOTE (Synthetic Minority Oversampling Technique) and ADASYN (Adaptive Synthetic Sampling).
- **Model Comparison**: Train and evaluate multiple models, including Logistic Regression, XGBoost, Decision Trees, and Random Forest.
- **Performance Evaluation**: Assess models based on accuracy, sensitivity, specificity, F1-score, and ROC-AUC.
- **Visualization**: Generate Precision-Recall Curves and ROC Curves for model performance comparison.

## Models Used

1. **Logistic Regression**
2. **XGBoost**
3. **Decision Trees**
4. **Random Forest**
5. **Logistic Regression with SMOTE**
6. **XGBoost with SMOTE**
7. **Random Forest with SMOTE**
8. **Logistic Regression with ADASYN**
9. **XGBoost with ADASYN**
10. **Decision Trees with ADASYN**

## Technologies and Libraries

- **Python**
- **Scikit-learn**: For implementing machine learning algorithms and metrics
- **XGBoost**: For Gradient Boosting algorithms
- **Imbalanced-learn**: For resampling techniques like SMOTE and ADASYN
- **Matplotlib** & **Seaborn**: For plotting graphs and visualizing results
- **Pandas** & **NumPy**: For data manipulation and analysis

## Dataset

The dataset contains a collection of credit card transaction records. Each record includes information about the transaction and whether it was fraudulent or not. The dataset is highly imbalanced, with fraudulent transactions being a minority class.

## Steps for Running the Project

1. **Clone the Repository**

   Clone the project repository to your local machine:

   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   ```

2. **Install Dependencies**

   Create a virtual environment and install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**

   The main notebook for the project is `credit_card_fraud_detection.ipynb`. Open it and run the cells in order to execute the entire pipeline.

   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

4. **Model Training and Evaluation**

   The notebook contains sections for data preprocessing, model training, and evaluation. Follow the steps in the notebook to train and evaluate the models.

## Performance Metrics

| Model                            | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |
|----------------------------------|----------|----------------------|-------------|----------|---------|
| Logistic Regression              | 99.93%   | 0.6497               | 0.9999      | 0.7518   | High    |
| Logistic Regression with SMOTE   | 94.46%   | 91.79%               | 97.13%      | 94.31%   | High    |
| Random Forest                    | 99.97%   | 0.8299               | 0.9999      | 0.9058   | High    |
| Random Forest with SMOTE         | 100%     | 1.00                 | 1.00        | 1.00     | High    |
| XGBoost                          | 100%     | 1.00                 | 1.00        | 1.00     | High    |
| XGBoost with SMOTE               | 100%     | 1.00                 | 0.9999      | 1.00     | High    |
| Logistic Regression with ADASYN  | 88.22%   | 86.50%               | 89.95%      | 88.02%   | 0.92    |
| XGBoost with ADASYN              | 99.99%   | 1.00                 | 0.99998     | 0.9999   | 1.00    |
| Decision Trees                   | 99.02%   | 99.99%               | 92.74%      | 0.9903   | 1.00    |
| Decision Trees with ADASYN       | 99.02%   | 92.04%               | 97.87%      | 0.13     | 0.96    |

## Results Summary

- **Best Performing Models**: Logistic Regression with SMOTE and Random Forest with SMOTE provided the best performance in terms of F1-Score, Sensitivity, Specificity, and ROC-AUC.
- **Class Imbalance Handling**: Techniques like SMOTE and ADASYN were crucial in improving model performance on imbalanced datasets.

## Conclusion

The project demonstrates the successful application of machine learning algorithms to detect fraudulent credit card transactions. **Logistic Regression with SMOTE** and **Random Forest with SMOTE** stood out as the most effective models, balancing accuracy and recall. This model is capable of identifying fraudulent transactions with high precision, making it a strong candidate for deployment in real-world financial systems.

## Future Work

- Experiment with more advanced algorithms (e.g., Neural Networks) for potential performance gains.
- Implement real-time fraud detection systems.
- Perform additional feature engineering to improve model performance further.


## Contributors
  - Zeba Firdouse
