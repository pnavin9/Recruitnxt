import pandas as pd
import re
from docx import Document
from pathlib import Path


class ResumeProcessor:
    """
    Class to process resumes and extract information.

    Args:
    - folder_path (str): Path to the folder containing resume files.

    Attributes:
    - folder_path (Path): Path object representing the folder containing resume files.
    """

    def __init__(self, folder_path):
        current_path = Path.cwd()  # Get the current working directory
        self.folder_path = current_path / folder_path  # Combine paths using Path objects

    def extract_text(self, filename):
        """
        Extract text from a resume file.

        Args:
        - filename (str): Name of the resume file without extension.

        Returns:
        - str: Extracted text from the resume file.
        """
        try:
            filepath = self.folder_path.joinpath(filename.upper() + " Resume.docx")
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error occurred while processing {filepath}: {e}")
            return None

    def clean_text(self, text):
        """
        Clean extracted text by removing non-alphanumeric characters, Twitter handles, and extra whitespace.

        Args:
        - text (str): Text to be cleaned.

        Returns:
        - str: Cleaned text.
        """
        try:
            # Remove Twitter handles starting with '@'
            text = re.sub(r"@\w+", "", text)
            # Remove non-alphanumeric characters and extra whitespace
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            # Convert multiple whitespace characters to a single space
            text = re.sub(r"\s+", " ", text)
            # Convert the text to lowercase
            text = text.lower()
            return text
        except Exception as e:
            print(f"Error occurred while processing: {e}")
            return None

    def find_skills(self, text, skills):
        """
        Find skills mentioned in the text.

        Args:
        - text (str): Text to search for skills.
        - skills (pandas.DataFrame): DataFrame containing skills to search for.

        Returns:
        - list: List of skills found in the text.
        """
        found_skills = []
        for skill_name in skills["skill_name"]:
            if str(skill_name).lower() in str(text).lower():
                found_skills.append(skill_name)
        return found_skills

    def process_dataframe(self, df):
        """
        Process a DataFrame containing candidate information.

        Args:
        - df (pandas.DataFrame): DataFrame containing candidate information.

        Returns:
        - tuple: Tuple containing language DataFrame and skill DataFrame.
        """
        df["Extracted_Text"] = df["CandidateID"].apply(self.extract_text)
        df["Extracted_Text"] = df["Extracted_Text"].apply(self.clean_text)

        # Create DataFrame for languages
        language_columns = [
            'assamese', 'bengali', 'gujarati', 'hindi', 'kannada', 'kashmiri', 'konkani',
            'malayalam', 'manipuri', 'marathi', 'nepali', 'oriya', 'punjabi', 'sanskrit', 'english',
            'sindhi', 'tamil', 'telugu', 'urdu', 'bodo', 'santhali', 'maithili', 'dogri'
        ]
        language_df = pd.DataFrame(columns=["CandidateID"] + language_columns)
        language_df["CandidateID"] = df["CandidateID"]

        # Check for language proficiency in each language
        for language in language_df.columns[1:]:
            language_df[language] = df["Extracted_Text"].apply(
                lambda x: 1 if language.lower() in str(x).lower() else 0
            )

        # Create DataFrame for skills
        skill_df = pd.DataFrame(columns=["CandidateID", "Skill", "Skill_count"])
        skill_df["CandidateID"] = df["CandidateID"]

        # Read skills from file
        skill = pd.read_csv("rx_skills.csv")

        # Find skills in each resume
        skill_df["Skill"] = [self.find_skills(text, skill) for text in df["Extracted_Text"]]
        skill_df["Skill_count"] = skill_df["Skill"].apply(len)

        return language_df, skill_df
