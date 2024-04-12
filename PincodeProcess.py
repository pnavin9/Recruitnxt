import indiapins
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
from tqdm import tqdm
import numpy as np

class PincodeDetailsExtractor:
    """
    A class to extract details and calculate distance between two pincodes in India.
    """

    def __init__(self):
        """
        Initialize PincodeDetailsExtractor class with an empty DataFrame.
        """
        self.pincode_details_df = pd.DataFrame(columns=['R_District', 'R_Region', 'R_State', 'B_District', 'B_Region', 'B_State', 'Distance_Kilometers'])

    def get_pincode_details(self, pincode):
        """
        Retrieve details (district, region, state) for a given pincode.
        
        Parameters:
            pincode (str): The pincode for which details are to be retrieved.
        
        Returns:
            dict or None: A dictionary containing details of the pincode, or None if details are not found.
        """
        try:
            details = indiapins.matching(pincode)
            if details:
                return details[0]  # Return the details of the first matching pincode
            else:
                return None
        except Exception as e:
            print(f"Error occurred while getting details for pincode {pincode}: {e}")
            return None

    def get_coordinates(self, pincode):
        """
        Retrieve latitude and longitude coordinates for a given pincode.
        
        Parameters:
            pincode (str): The pincode for which coordinates are to be retrieved.
        
        Returns:
            tuple or None: A tuple containing latitude and longitude coordinates, or None if coordinates are not found.
        """
        geolocator = Nominatim(user_agent="pincode_locator")
        try:
            location = geolocator.geocode(pincode + ", India")
            if location:
                return location.latitude, location.longitude
            else:
                return None
        except Exception as e:
            print(f"Error occurred while getting coordinates for pincode {pincode}: {e}")
            return None

    def calculate_distance(self, pincode1, pincode2):
        """
        Calculate distance (in kilometers) between two pincodes.
        
        Parameters:
            pincode1 (str): The first pincode.
            pincode2 (str): The second pincode.
        
        Returns:
            float or None: The distance between the two pincodes in kilometers, or None if calculation fails.
        """
        coords1 = self.get_coordinates(pincode1)
        coords2 = self.get_coordinates(pincode2)
        if coords1 and coords2:
            return geodesic(coords1, coords2).kilometers
        else:
            return None

    def process_data(self, df):
        """
        Process DataFrame containing residential and branch pincode information.
        Extract details for each pincode, calculate distance between residential and branch, and store in DataFrame.
        
        Parameters:
            df (DataFrame): DataFrame containing columns 'R_Pincode', 'B_Pincode', and 'CandidateID'.
        """
        for index, row in tqdm(df.iterrows(), total=len(df)):
            residential_pincode = None
            branch_pincode = None
            if row['R_Pincode'] and not np.isnan(row['R_Pincode']):
                residential_pincode = str(int(row['R_Pincode']))
            if row['B_Pincode'] and not np.isnan(row['B_Pincode']):
                branch_pincode = str(int(row['B_Pincode']))
            candidate_id = row['CandidateID']
            residential_details = self.get_pincode_details(residential_pincode)
            branch_details = self.get_pincode_details(branch_pincode)

            if not residential_details:
                residential_details = {
                    'District': 'Unknown',
                    'Region': 'Unknown',
                    'State': 'Unknown'
                }

            if not branch_details:
                branch_details = {
                    'District': 'Unknown',
                    'Region': 'Unknown',
                    'State': 'Unknown'
                }

            distance = self.calculate_distance(residential_pincode, branch_pincode)
            if not distance:
                distance = -1
        
            new_row = pd.Series({
                'CandidateID': candidate_id,
                'R_District': residential_details['District'].lower(),
                'R_Region': residential_details['Region'].lower(),
                'R_State': residential_details['State'].lower(),
                'B_District': branch_details['District'].lower(),
                'B_Region': branch_details['Region'].lower(),
                'B_State': branch_details['State'].lower(),
                'Distance_Kilometers': distance
            })

            self.pincode_details_df = pd.concat([self.pincode_details_df, pd.DataFrame([new_row])], ignore_index=True)

    def get_processed_data(self):
        """
        Get the processed DataFrame containing extracted details and calculated distances.
        
        Returns:
            DataFrame: Processed DataFrame containing columns 'CandidateID', 'R_District', 'R_Region', 'R_State', 'B_District', 'B_Region', 'B_State', 'Distance_Kilometers'.
        """
        return self.pincode_details_df
