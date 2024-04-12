import pandas as pd
import re

class DistrictDataProcessor:
    """
    Class to process district demographics data and merge it with other data.

    Args:
    - district_file_path (str, optional): Path to the district demographics CSV file. Defaults to '.district_demographics.csv'.

    Attributes:
    - district_file_path (str): Path to the district demographics CSV file.
    - districts (pandas.DataFrame): DataFrame containing district demographics data.
    """

    def __init__(self, district_file_path='.district_demographics.csv'):
        self.district_file_path = district_file_path
        self.districts = None

    def load_district_data(self):
        """
        Load district demographics data from CSV file and perform preprocessing.

        """
        # Read the district data from CSV
        self.districts = pd.read_csv(self.district_file_path)
        # Drop unnecessary column
        self.districts.drop(columns=['Unnamed: 1'], inplace=True)
        # Drop rows with NaN values
        self.districts.dropna(inplace=True)
        # Convert strings to lowercase
        self.districts = self.districts.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        # Clean population, area, and density columns
        self.districts['Population'] = self.districts['Population'].apply(self.extract_number)
        self.districts['Area'] = self.districts['Area'].apply(self.extract_number)
        self.districts['Density'] = self.districts['Density'].apply(self.extract_number)
        
    def extract_number(self, string):
        """
        Extract numeric values from a string.

        Args:
        - string (str): String containing numeric values.

        Returns:
        - int or float: Extracted numeric value, or None if not found.
        """
        try:
            num = float(string)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            # Remove commas and square brackets
            string = string.replace(',', '')
            # Remove any extra text enclosed in square brackets
            string = re.sub(r'\[.*?\]', '', string)
            # Remove commas
            string = string.replace(',', '')
            # Try converting to integer, return None if not possible
            try:
                return int(string)
            except ValueError:
                return None

    def merge_district_data(self, df):
        """
        Merge district demographics data with another DataFrame.

        Args:
        - df (pandas.DataFrame): DataFrame to merge with district demographics data.

        Returns:
        - pandas.DataFrame: Merged DataFrame.
        """
        if self.districts is None:
            print("District data not loaded. Please call load_district_data() first.")
            return

        # Merge data on R_District
        df = pd.merge(df, self.districts, how='left', left_on='R_District', right_on='Ddistrict')
        df.rename(columns={'Population': 'R_Population', 'Density': 'R_Density'}, inplace=True)
        df.drop(columns=['Ddistrict', 'Area'], inplace=True)

        # Merge data on B_District
        df = pd.merge(df, self.districts, how='left', left_on='B_District', right_on='Ddistrict')
        df.rename(columns={'Population': 'B_Population', 'Density': 'B_Density'}, inplace=True)
        df.drop(columns=['Ddistrict', 'Area'], inplace=True)

        return df
