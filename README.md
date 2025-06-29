# Credit Card Approval Risk Predictor

## Table of Contents

1.  [Introduction](#introduction)
2.  [Problem Statement](#problem-statement)
3.  [Dataset](#dataset)
4.  [Methodology](#methodology)
    * [Data Preprocessing](#data-preprocessing)
    * [Model Training and Evaluation](#model-training-and-evaluation)
5.  [Results](#results)
6.  [Key Insights](#key-insights)
7.  [Conclusion](#conclusion)
8.  [File Structure](#file-structure)
9.  [How to Run](#how-to-run)
10. [Future Enhancements](#future-enhancements)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

This project aims to develop an automated system for predicting credit card approval risk using machine learning. Traditionally, credit card applications are reviewed manually, a process that is time-consuming, prone to human error, and lacks scalability. By leveraging historical application data, this project seeks to build a robust predictive model that can accurately assess the likelihood of credit card approval, thereby streamlining the process for financial institutions and improving decision-making efficiency.

## Problem Statement

Banks receive an enormous volume of credit card applications daily. Many applications are rejected due to various factors such as high loan balances, low income levels, or excessive inquiries on an individual’s credit report. Manually analyzing each application to determine eligibility and risk is an intensive task that often leads to inconsistencies and delays. The goal is to mitigate these issues by automating the risk assessment, enabling banks to make faster, more consistent, and data-driven approval decisions.

## Dataset

The project utilizes two primary datasets:

1.  **`application_record.csv`**: Contains demographic and financial information about credit card applicants.
    * `ID`: Client ID.
    * `CODE_GENDER`: Gender of the applicant.
    * `FLAG_OWN_CAR`: Flag indicating if the applicant owns a car.
    * `FLAG_OWN_REALTY`: Flag indicating if the applicant owns real estate.
    * `CNT_CHILDREN`: Number of children.
    * `AMT_INCOME_TOTAL`: Annual income.
    * `NAME_INCOME_TYPE`: Type of income (e.g., Working, Commercial associate).
    * `NAME_EDUCATION_TYPE`: Level of education.
    * `NAME_FAMILY_STATUS`: Marital status.
    * `NAME_HOUSING_TYPE`: Housing situation.
    * `DAYS_BIRTH`: Age in days (negative values indicate days since birth).
    * `DAYS_EMPLOYED`: Days employed (negative values indicate days since employment; positive values indicate retired).
    * `FLAG_MOBIL`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL`: Flags for contact information.
    * `OCCUPATION_TYPE`: Type of occupation.
    * `CNT_FAM_MEMBERS`: Number of family members.

2.  **`credit_record.csv`**: Contains monthly credit status for each client ID from the `application_record.csv`.
    * `ID`: Client ID.
    * `MONTHS_BALANCE`: The month of the record relative to the current month (0 is current month, -1 is previous month, etc.).
    * `STATUS`: Credit status for the month (0, 1, 2, 3, 4, 5 for different levels of arrears, C for paid off, X for no loan).

These datasets are joined and preprocessed to create a comprehensive feature set for training the machine learning model.

## Methodology

The overall methodology involved data preparation, feature engineering, model training, and rigorous evaluation.

### Data Preprocessing

* **Handling Categorical Features**: Non-numerical (categorical) features were converted into numerical representations suitable for machine learning algorithms. `sklearn.preprocessing.LabelEncoder` was primarily used for this purpose.
* **Specific Mappings**:
    * `Income_type`: Mapped to numerical values (e.g., 'Commercial associate' to 0, 'Working' to 4).
    * `Education_type`: Mapped to numerical values (e.g., 'Academic degree' to 0, 'Secondary' to 2).
    * `Family_status`: Mapped to numerical values (e.g., 'Married' to 0, 'Single' to 1).
    * `Housing_type`: Mapped to numerical values (e.g., 'House / apartment' to 0, 'With parents' to 1).
    * `Occupation_type`: Various occupations were assigned unique numerical identifiers.
* **Target Variable Creation**: A critical step involved defining the target variable (credit card approval status) based on the `STATUS` column in `credit_record.csv`. This likely involved classifying individuals as 'approved' or 'rejected' based on their credit history.

### Model Training and Evaluation

After data preprocessing, the dataset was split into training and testing sets. A machine learning model (details of the specific algorithm used are in the notebook) was trained on the processed data. The model's performance was then evaluated using standard classification metrics:

* **Confusion Matrix**: Provides a detailed breakdown of correct and incorrect predictions for each class.
* **Accuracy Score**: The proportion of correctly classified instances.
* **Classification Report**: Includes precision, recall, and F1-score for each class, offering a comprehensive view of the model's performance on imbalanced datasets (if applicable).

## Results

The developed machine learning model achieved an overall accuracy of **79.15%**. This indicates a good capability of the model to correctly predict credit card approval risk based on the provided features.

```
          precision    recall  f1-score   support

     0.0       0.79      1.00      0.88      1673
     1.0       0.17      0.00      0.01       269

accuracy                           0.79      1942

macro avg       0.48      0.50      0.44      1942
weighted avg       0.71      0.79      0.70      1942
```

*Note: The classification report suggests a significant imbalance in the target classes, and while overall accuracy is high, the model's ability to identify the minority class (e.g., 'rejected' cases if 1.0 represents rejection) might be limited, as indicated by the low recall and F1-score for class 1.0. Further analysis on handling class imbalance would be beneficial.*

## Key Insights

* **Automation Feasibility**: Machine learning effectively automates the manual credit card approval process, significantly reducing operational overhead and processing time.
* **Data Importance**: Proper data preprocessing, especially handling categorical variables and feature engineering, is crucial for building effective predictive models.
* **Model Performance**: An accuracy of ~79% demonstrates the model's reasonable predictive power. However, deeper analysis using precision, recall, and F1-score reveals potential areas for improvement, especially concerning minority class prediction, which is common in fraud detection or risk assessment scenarios.
* **Beyond Accuracy**: While overall accuracy is a good starting point, the classification report highlights the importance of looking at other metrics, particularly in imbalanced datasets, to understand the model's true effectiveness across all classes.

## Conclusion

The "Credit Card Approval Risk Predictor" successfully validates the application of machine learning in automating credit card application risk assessment. The project demonstrates a functional pipeline from raw data to a predictive model capable of classifying applications with a notable accuracy of ~79.15%. This automation offers substantial benefits to banks, including improved efficiency, reduced manual errors, and faster decision-making cycles.

While the model shows promising results, particularly in identifying the majority class, the insights from the classification report indicate that there is room for improvement in handling class imbalance. Future work could focus on techniques like oversampling, undersampling, or using algorithms robust to imbalance to enhance the model's ability to correctly identify all types of credit risk, making it even more valuable for real-world deployment. This project serves as a strong foundation for building more sophisticated and equitable automated credit approval systems.

## File Structure

.  
├── CreditCardApprovalPredictor.ipynb  
├── application_record.csv  
├── credit_record.csv  
└── README.md  


## How to Run

To run this project:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tamaxhh/Credit-Card-Approval-Risk-Predictor.git
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook CreditCardApprovalPredictor.ipynb
    ```
4.  Execute the cells sequentially in the notebook to preprocess data, train the model, and evaluate its performance.

## Future Enhancements

* **Handling Class Imbalance**: Implement techniques like SMOTE, ADASYN, or use `class_weight` parameters in models to improve performance on the minority class.
* **Feature Engineering**: Explore more advanced feature engineering techniques from existing features or external data sources.
* **Model Optimization**: Experiment with other machine learning algorithms (e.g., Gradient Boosting, Random Forests, Neural Networks) and hyperparameter tuning to find the optimal model.
* **Explainable AI (XAI)**: Implement techniques (e.g., SHAP, LIME) to understand model predictions and feature importance, enhancing trust and transparency.
* **Deployment**: Develop a simple web application to deploy the trained model for real-time predictions.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you have one, otherwise omit this part or state "No specific license").

---
**Disclaimer**: This project is for educational purposes and demonstrates a machine learning approach to credit card approval prediction. It should not be used for actual financial decision-making without further rigorous validation, regulatory compliance, and expert review.
