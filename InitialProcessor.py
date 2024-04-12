import pandas as pd
import numpy as np

class InitialProcessor:
    """
    A class for preprocessing data in a DataFrame for analysis.

    Attributes:
    df (DataFrame): The DataFrame containing the data to be processed.

    Methods:
    __init__: Initializes the InitialProcessor object.
    rename_columns: Renames columns in the DataFrame.
    separate_products: Separates products into binary columns in the DataFrame.
    clean_ticketsize: Cleans the 'Ticket_Size' column in the DataFrame.
    clean_incentive: Cleans the 'Incentive' column in the DataFrame.
    clean_familymembers: Cleans the 'Family_Members' column in the DataFrame.
    format_doj: Formats the 'DOJ' (Date of Joining) column to datetime format.
    clean_organizations: Cleans the 'Previous_Organizations' column in the DataFrame.
    reduce_earning_members: Reduces 'Earning_Members' values to single digits.
    """

    def __init__(self, df):
        """
        Initializes the InitialProcessor object.

        Args:
        df (DataFrame): The DataFrame containing the data to be processed.
        """
        def lowercase_str(x):
            """
            Converts string values to lowercase.

            Args:
            x: Value to be converted.

            Returns:
            Converted lowercase string or original value if not a string.
            """
            if isinstance(x, str):
                return x.lower()
            else:
                return x
        # Apply the function element-wise to the entire DataFrame
        df = df.applymap(lowercase_str)
        self.df = df

    def rename_columns(self):
        """
        Renames columns in the DataFrame according to a predefined mapping.
        """
        new_column_names = {
            'Have you Completed your Graduation ?': 'Graduation',
            'Highest Educational Qualification': 'Qualification',
            'Total no of years Experience [before joining Piramal]' : 'Experience',
            'Previous Industry worked with [before joining Piramal]' : 'Industry',
            'Name of your Previous Organization / Company' : 'Company',
            'How many Organization that you have worked before joining Piramal Finance ?' : 'Previous_Organizations',
            'Average Incentive [per month] earned in your pervious company ?' : 'Incentive',
            'How did you come to know about the role at Piramal Finance ?' : 'Source',
            'Which Products you are selling in your pervious role ?' : 'Products',
            'What was the average ticket size handled at your end in previous role ?' : 'Ticket_Size',
            'How many members are there in your family ?' : 'Family_Members',
            'How many are earning family members ? [Other then yourself]2' : 'Earning_Members',
            'How many members are dependent on you ?' : 'Dependent_Members',
            'Department' : 'Department',
            'DOJ' : 'DOJ',
            'Location Code' : 'Location',
            'Residential Pincode' : 'R_Pincode',
            'Branch Pincode' : 'B_Pincode',
            'Performance' : 'Performance'
        }
        self.df.rename(columns=new_column_names,inplace = True)
        
    def separate_products(self):
        """
        Separates products into binary columns in the DataFrame.
        """
        columns_to_add = ['current / saving account [casa]',
                    'current account â\x80\x93 saving account/others',
                    'msme / sme loan',
                    'housing loan',
                    'others',
                    'used car loan',
                    'personal loan',
                    'fmcg',
                    'loan against property/ secured business loan',
                    'car loan / used car loan',
                    'unsecure business loan']
        
        # Function to set value to 1 if the column name is present in Products else 0
        def set_value(column_name, products):
            """
            Sets value to 1 if the column name is present in 'Products', else 0.

            Args:
            column_name (str): The name of the column.
            products (str): String containing products.

            Returns:
            1 if column name is present in products, else 0.
            """
            if pd.notna(products) and column_name in products:
                return 1
            else:
                return 0

        # Create new columns based on the list of column names and set their values
        for column_name in columns_to_add:
            self.df[column_name] = self.df['Products'].apply(lambda x: set_value(column_name, x))
        
        # Drop the Products column
        self.df = self.df.drop(columns=['Products'])

    def clean_ticketsize(self):
        """
        Cleans the 'Ticket_Size' column in the DataFrame.
        """
        value_mapping = {
        'inr 5l - inr 15l': 1000000,  
        'inr 50k - inr 2l': 150000,   
        'inr 15l and above': 2000000, 
        'fresher': 0,                 
        'inr 10l and above': 1500000, 
        '\xa0inr 2l - inr 5l': 350000,
        'åêinr 2l - inr 5l' : 350000,
        'inr 50k and below': 50000,   
        'inr 5l - inr 10l': 750000    
        }

        # Replace values in the 'ticket size' column using the mapping
        self.df['Ticket_Size'] = self.df['Ticket_Size'].replace(value_mapping)

    def clean_incentive(self):
        """
        Cleans the 'Incentive' column in the DataFrame.
        """
        value_mapping = {
        'above 10k': 12000,  
        '7k-10k': 8000,    
        'less than 3k': 20000,  
        'nil': 0,                  
        '3k-7k': 5000,  
        }

        # Replace values in the 'ticket size' column using the mapping
        self.df['Incentive'] = self.df['Incentive'].replace(value_mapping)

    def clean_familymembers(self):
        """
        Cleans the 'Family_Members' column in the DataFrame.
        """
        value_mapping = {
        '1 - 2 members': 2,  
        '3 - 4 members': 4,    
        '5 & above members': 6,  
        }

        # Replace values in the 'ticket size' column using the mapping
        self.df['Family_Members'] = self.df['Family_Members'].replace(value_mapping)
        
    def format_doj(self):
        """
        Formats the 'DOJ' (Date of Joining) column to datetime format 
        and adds a new column 'Days_passed' with the difference between DOJ 
        and current date.
        """
        try:
            # Try formatting with specific format (%d-%m-%Y)
            self.df['DOJ'] = pd.to_datetime(self.df['DOJ'], format='%d-%m-%Y')
        except ValueError:
            # If format fails, try using 'mixed' format
            self.df['DOJ'] = pd.to_datetime(self.df['DOJ'], format='mixed')

        # Get today's date
        today = pd.Timestamp('today')

        # Calculate the difference in days for each row in 'DOJ'
        self.df['Days_passed'] = (today - self.df['DOJ']).dt.days

    def clean_organizations(self):
        """
        Cleans the 'Previous_Organizations' column in the DataFrame.
        """
        value_mapping = {
        '01-feb': 2,  
        '03-may': 4,    
        '0 / fresher': 0,  
        '5+': 6,  
        }

        # Replace values in the 'ticket size' column using the mapping
        self.df['Previous_Organizations'] = self.df['Previous_Organizations'].replace(value_mapping)
        
    def reduce_earning_members(self):
        """
        Reduces 'Earning_Members' values to single digits.
        """

        def take_first_digit(x):
            """
            Returns the first digit of a number.

            Args:
            x: The number.

            Returns:
            The first digit of the number.
            """
            if float(x) > 10:
                return int(str(x)[0])
            else:
                return float(x)

        # Apply the custom function to the 'Numbers' column
        self.df['Earning_Members'] = self.df['Earning_Members'].apply(take_first_digit)
