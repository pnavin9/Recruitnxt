import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re
import pickle
import json

class Preprocessor:
    """
    Class to preprocess data for machine learning models.

    Attributes:
    - value_mapping (dict): Mapping of values to be replaced in the 'Previous_Organizations' column.
    - states (list): List of Indian states.
    """

    def __init__(self):
        self.value_mapping = {
            'a': 6,  # Example: 10 lakhs
        }

        self.states = ['andhra pradesh', 'assam', 'bihar', 'chandigarh', 'chhattisgarh', 'delhi', 'gujarat', 'haryana',
                       'himachal pradesh', 'jammu and kashmir', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh',
                       'maharashtra', 'odisha', 'puducherry', 'punjab', 'rajasthan', 'tamil nadu', 'telangana', 'unknown',
                       'uttar pradesh', 'uttarakhand', 'west bengal']

    def preprocess(self, df):
        """
        Preprocesses the input DataFrame for machine learning.

        Args:
        - df (pandas.DataFrame): Input DataFrame.

        Returns:
        - pandas.DataFrame: Preprocessed DataFrame.
        """

        # Replace values in 'Previous_Organizations' column
        df['Previous_Organizations'] = df['Previous_Organizations'].replace(self.value_mapping)

        # Drop specified columns
        df.drop(columns=['Designation', 'DOJ', 'R_District', 'B_District', 'Skill'], inplace=True)

        # Replace Graduation and Qualification values
        df['Graduation'] = df['Graduation'].fillna(0).map({'full time': 1, 'part time': 0, 0: 0})
        df['Qualification'] = df['Qualification'].map(
            {'graduate': 2, 'under graduate': 1, 'post graduate': 3, 'others': 0, 'diploma holders': 4}).fillna(-1)

        # Set 'Previous_Organizations' as type int, treating NaN as 0
        df['Previous_Organizations'] = df['Previous_Organizations'].fillna(0).astype(int)

        # One-hot encoding for categorical variables
        columns_to_encode = ['Industry', 'Source', 'Department', 'R_Region', 'B_Region']
        df_encoded = pd.get_dummies(df, columns=columns_to_encode)

        # Create indicator variables for states
        for state in self.states:
            df_encoded[state] = ((df['R_State'] == state) | (df['B_State'] == state)).astype(int)

        # Drop 'R_State' and 'B_State'
        df_encoded.drop(columns=['R_State', 'B_State'], inplace=True)

        # Modify column names to ensure compatibility
        df_encoded.columns = [re.sub(r'\W+', '_', col) for col in df_encoded.columns]

        # Load features from pickle file
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)

        # Get missing columns
        missing_columns = list(set(features) - set(df_encoded.columns))

        # Fill missing columns with zeros
        df_encoded[missing_columns] = 0

        # Load column info from JSON file
        with open("column_info.json", "r") as f:
            column_info = json.load(f)

        # Convert columns to specified data types
        for col, dtype in column_info.items():
            df_encoded[col] = df_encoded[col].astype(dtype)

        return df_encoded
