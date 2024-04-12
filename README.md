# Recruitnxt

This project aims to automate the recruitment process for Piramal, a finance firm that lends loans to various target customers, including small retail owners. The project involves processing candidate data, analyzing resumes, and training machine learning models to predict candidate performance.

## Overview

Piramal's recruitment process involves hiring salespersons to sell loans to potential customers. As the business grows, there is a need to automate the hiring process based on candidate CVs and responses to certain questions. This project aims to streamline this process through automated data processing and model training.

## Steps

### Step 0: Understanding the Problem Statement

Before diving into the technical details, it's essential to understand Piramal's requirements and objectives. This step involves understanding the recruitment process, identifying key factors for candidate evaluation, and defining the scope of the project.

### Step 1: Data Preprocessing

Data preprocessing is a crucial step in preparing the candidate data for analysis and model training. This step involves several sub-steps:

- **Renaming Columns and Fixing Typos**: Renaming columns for convenience and fixing typos in column names and categorical values.
- **Fixing Categorical Columns**: Using ordinal encoding for mapping categorical variables like ticket size, family members, and incentives, and one-hot encoding for columns where order doesn't matter.
- **Using Pincodes**: Utilizing pincode data to extract demographic information such as district, state, population, and density.
- **Extracting Company Information**: Extracting company information from provided links to enhance candidate profiles.
- **Parsing Resumes**: Extracting language proficiency and skills from candidate resumes and preprocessing the text data.

### Step 2: Exploratory Data Analysis (EDA)

EDA is a critical step in understanding the characteristics of the candidate data and identifying useful features for analysis and modeling. This step involves visualizing data distributions, exploring relationships between variables, and identifying patterns and trends.

### Step 3: Model Training and Hyperparameter Tuning

Model training involves building predictive models to classify candidate performance based on available features. This step includes:

- **Base Model Evaluation**: Evaluating performance using AutoML and LazyClassifier to assess various machine learning algorithms.
- **Model Selection**: Selecting suitable models based on performance evaluation.
- **Hyperparameter Tuning**: Fine-tuning model hyperparameters using techniques like grid search or Optuna to optimize model performance.

### Step 4: Inference

Inference involves applying the trained model to new candidate data to make predictions about candidate performance. This step includes evaluating model performance on test data and generating predictions.

### Step 5: Further Scope

Identifying areas for further improvement and refinement in the recruitment process. This step involves proposing enhancements to the questionnaire, exploring additional features for candidate evaluation, and refining the model based on feedback and performance evaluation.

## Files Description

1. **InitialProcessor.py**: Rename columns, fix typos, and perform initial data cleaning.
2. **PincodeProcess.py**: Extract pincode details and demographic information.
3. **Demographics.py**: Process demographic data and merge with candidate data.
4. **Companies.py**: Extract company information from provided links.
5. **CVManual.py**: Parse resumes to extract language proficiency and skills.
6. **FinalProcessing.py**: Perform final preprocessing steps for model training.
7. **Train.py**: Train machine learning models and optimize hyperparameters.
8. **main.py**: Main script to execute the entire pipeline.

## Usage

1. **Installation**: Install the required dependencies using the following command:
    ```
    pip install -r requirements.txt
    ```

2. **Data Preprocessing and Model Training**:
    ```
    python main.py -f <folder containing resumes> True
    ```
    - Replace `<folder containing resumes>` with the path to the folder containing candidate resumes.
    - This command initiates data preprocessing, model training, and hyperparameter tuning. Set the last argument to `True` for training.

3. **Prediction**:
    ```
    python main.py -f <folder containing resumes> False
    ```
    - Replace `<folder containing resumes>` with the path to the folder containing candidate resumes.
    - This command makes predictions using the trained model. Set the last argument to `False` for inference/prediction.

Note: You might need to change some paths because some of the required files are present in DataScource

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- XGBoost
- chardet
- docx
- getopt
- pickle
- numpy

## Further Information

For more details on the project and its implementation, refer to the source code files and accompanying documentation.

## Author

Navin Patwari

## Contact

For any inquiries or support, please contact [patwarinavin9@gmail.com].
