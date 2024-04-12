import pandas as pd
import sys
import getopt
import chardet
from InitialProcessor import InitialProcessor
from PincodeProcess import PincodeDetailsExtractor
from Demographics import DistrictDataProcessor
from Companies import CompanyScraper
from CVManual import ResumeProcessor
from FinalProcessing import Preprocessor
from Train import ModelTrainer
import pickle
import os
import numpy as np
import xgboost

def load_data(argv):
    """
    Load data from a CSV file.

    Args:
    - argv (list): Command-line arguments.

    Returns:
    - pandas.DataFrame: Loaded DataFrame.
    """
    inputfile = ''
    try:
        opts, _ = getopt.getopt(argv, "hf:", ["file="])
    except getopt.GetoptError:
        print('Usage: python script.py -f <filename>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python script.py -f <filename>')
            sys.exit()
        elif opt in ("-f", "--file"):
            inputfile = arg

    if inputfile == '':
        print('Usage: python script.py -f <filename>')
        sys.exit(2)

    # Detect the encoding of the file
    with open(inputfile, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    # Load the data with the detected encoding
    df = pd.read_csv(inputfile, encoding=encoding)

    return df[:15]

def main(argv):
    """
    Main function to execute data processing steps.

    Args:
    - argv (list): Command-line arguments.
    """
    df = load_data(argv)

    # Initialize InitialProcessor object
    initial_processing = InitialProcessor(df)

    # Perform preprocessing steps
    initial_processing.rename_columns()
    initial_processing.separate_products()
    initial_processing.clean_ticketsize()
    initial_processing.clean_incentive()
    initial_processing.clean_familymembers()
    initial_processing.format_doj()
    initial_processing.clean_organizations()
    initial_processing.reduce_earning_members()

    df = initial_processing.df

    # Save the cleaned data
    cleaned_filename = 'cleaned_' + str(argv[1])
    df.to_csv(cleaned_filename, index=False)

    print(f"Cleaned data saved to {cleaned_filename}")
    
    print("Starting pincode processing...")

    pin_extractor = PincodeDetailsExtractor()
    # Process the data
    pin_extractor.process_data(df)
    # Get the processed data
    pin_data = pin_extractor.get_processed_data()
    
    df = pd.merge(df, pin_data, on='CandidateID', how='left')

    with_pincode_details = 'with_pincode_details_' + str(argv[1])
    df.to_csv(with_pincode_details, index=False)

    print(f"Data with pincode details saved to {with_pincode_details}")

    district_file_path = 'district_demographics.csv'
    processor = DistrictDataProcessor(district_file_path)
    processor.load_district_data()

    df = processor.merge_district_data(df)

    with_demographics = 'with_demographics_' + str(argv[1])
    df.to_csv(with_demographics, index=False)

    print(f"Data with demographics saved to {with_demographics}")

    print("Starting company data processing...")

    #company_processor = CompanyScraper("Company", link_columns=3, sleep_interval=5)
    #company_info = company_processor.scrape(df.copy())
    #df = pd.merge(df, company_info, on='Company', how='left')
    #df.drop(columns=['Company','Link1'], inplace=True)
    #df.rename(columns={'Link2':'Company_1', 'Link3':'Company_2'}, inplace=True)

    with_company_info = 'with_company_info_' + str(argv[1])
    df.to_csv(with_company_info, index=False)

    print(f"Data with company information saved to {with_company_info}")

    print("Starting resume processing...")
    
    folder_path = argv[2]
    processor = ResumeProcessor(folder_path)

    lang_df, skill_df = processor.process_dataframe(df.copy())

    df = pd.merge(df, lang_df, on='CandidateID', how='left')
    df = pd.merge(df, skill_df, on='CandidateID', how='left')

    df = df.drop_duplicates(subset=['CandidateID'], keep='first')

    cvmerged = 'cvmerged_' + str(argv[1])
    df.to_csv(cvmerged, index=False)

    print(f"Data with resume information saved to {cvmerged}")

    preprocessor = Preprocessor()

    # Preprocess data
    CandidateID = df['CandidateID']
    df = preprocessor.preprocess(df.copy())
    
    model = None

    # Check if training flag is provided
    if argv[3] == 'true':
        
        target_col = 'Performance'
        feature_cols = [col for col in df.columns if col not in ['CandidateID', 'Company', target_col]]

        # Create a ModelTrainer object
        trainer = ModelTrainer(train_data=df , target_col=target_col, feature_cols=feature_cols)

        trainer.optimize_hyperparams()

        model = trainer.train_final_model()  
        with open('xgboost_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print("Model saved as xgboost_model.pkl")
    else:
        # Load the model from the file
        with open('xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        target_col = 'Performance'
        with open('features.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        # Predict using the loaded model
        predictions = model.predict_proba(df[feature_cols])
        predictions = pd.DataFrame(predictions, columns=['Class_1', 'Class_2'])
        predictions['CandidateID'] = CandidateID

        predictions['Performance'] = np.where(predictions['Class_2'] > predictions['Class_1'], 1, 0)
        predictions['CandidateID'] = predictions['CandidateID'].str.upper()

        predictions.to_csv("predictions.csv", index=False)

        print("Predictions saved as predictions.csv")

if __name__ == "__main__":
    main(sys.argv[1:])
