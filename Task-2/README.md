# Loan Prediction 

## Overview
This project focuses on predicting loan approval status based on various applicant features. It involves data loading, cleaning, exploratory data analysis (EDA), and building machine learning models to classify loan applications as approved or not approved. An interactive tool is also provided to predict the loan status for new applications.

## Project Structure
The notebook covers the following key steps:
1.  **Data Loading**: Loading the `loan_prediction.csv` dataset.
2.  **Data Preprocessing**: Handling missing values, converting data types, and feature engineering.
3.  **Exploratory Data Analysis (EDA)**: Visualizing data distributions, relationships between features, and correlations.
4.  **Model Building**: Training and evaluating Decision Tree, Random Forest, and Logistic Regression classifiers.
5.  **Model Evaluation**: Comparing model performance using accuracy, classification reports, confusion matrices, and ROC curves.
6.  **Feature Importance**: Identifying key features influencing loan approval.
7.  **Interactive Prediction**: A function to predict loan status based on user-provided inputs.

## Dataset
The dataset used is `loan_prediction.csv`, which contains information about loan applicants, including:
-   `Loan_ID`: Unique Loan ID
-   `Gender`: Male/Female
-   `Married`: Yes/No
-   `Dependents`: Number of dependents (0, 1, 2, 3+)
-   `Education`: Graduate/Not Graduate
-   `Self_Employed`: Yes/No
-   `ApplicantIncome`: Applicant income
-   `CoapplicantIncome`: Co-applicant income
-   `LoanAmount`: Loan amount (in thousands)
-   `Loan_Amount_Term`: Term of loan in months
-   `Credit_History`: Credit history meets guidelines (1.0 = Yes, 0.0 = No)
-   `Property_Area`: Urban/Semi Urban/Rural
-   `Loan_Status`: Loan approved (Y/N)

## Methodology

### Data Cleaning and Preprocessing
-   **Missing Values**: 
    -   Categorical missing values (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`) were imputed using the mode.
    -   Numerical missing values (`LoanAmount`, `Loan_Amount_Term`) were imputed using `IterativeImputer` with `RandomForestRegressor`.
-   **Feature Engineering**: Created a `Total_Income` feature by summing `ApplicantIncome` and `CoapplicantIncome`, then dropped the original income columns.
-   **Categorical Encoding**: Converted categorical features (`Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`, `Dependents`) into numerical representations using `map` and `astype(int)`.

### Exploratory Data Analysis (EDA)
-   Count plots were generated for all categorical variables to understand their distribution.
-   Box plots were used to visualize the relationship between `Loan_Status` and `Total_Income`.
-   A correlation matrix heatmap was generated to show the relationships between numerical features.

### Model Training and Evaluation
-   The data was split into training and testing sets (70% train, 30% test).
-   Three classification models were trained:
    -   **Decision Tree Classifier**
    -   **Random Forest Classifier**
    -   **Logistic Regression**


## How to Use the Interactive Prediction Tool

1.  **Run all cells** in the notebook up to the `Enter Loan Application Details for Prediction` section.
2.  **Provide Inputs**: When prompted, enter the details for a new loan applicant (Gender, Married, Dependents, Education, Self Employed, Loan Amount, Loan Amount Term, Credit History, Property Area, Total Income).
3.  **Get Prediction**: The tool will output whether the loan application is "Approved" or "Not Approved" based on the trained Logistic Regression model.

### Example Usage:
```
Please provide the following details for loan approval prediction:
Gender (Male/Female): Male
Married (Yes/No): Yes
Dependents (0, 1, 2, or 3+): 0
Education (Graduate/Not Graduate): Graduate
Self Employed (Yes/No): Yes
Loan Amount (e.g., 150000): 66
Loan Amount Term in months (e.g., 360.0): 360
Credit History (1.0 for Yes, 0.0 for No): 1
Property Area (Urban/Rural/Semiurban): Urban
Total Income (Applicant Income + Coapplicant Income, e.g., 7000.0): 3000

Based on the provided details, the loan status is: Approved
```

## Libraries Used
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `sklearn`
