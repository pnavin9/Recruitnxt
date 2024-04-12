from googlesearch import search
from tqdm import tqdm
import pandas as pd
from urllib.parse import urlparse
import numpy as np
import time

class CompanyScraper:
    """
    This class scrapes information about companies from a given dataset.

    Attributes:
        data_column (str): Name of the column containing company names in the dataset.
        link_columns (int): Number of link columns to extract from search results (default: 3).
        sleep_interval (float): Time to wait between searches (default: 5 seconds).
        scrape_timeout (float): Maximum time to spend scraping a single company (default: 10 seconds).
    """

    def __init__(self, data_column, link_columns=3, sleep_interval=5, scrape_timeout=10):
        self.data_column = data_column
        self.link_columns = link_columns
        self.sleep_interval = sleep_interval
        self.scrape_timeout = scrape_timeout

    def scrape(self, df):
        """
        Scrapes information about companies from the provided DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing company names.

        Returns:
            pandas.DataFrame: A new DataFrame with scraped company information.
        """

        companies = df[self.data_column].unique()
        data_rows = []

        for company in tqdm(companies):
            start_time = time.time()
            search_results = [company] + [
                result for result in search(f"Wikipedia {company} India", num_results=1, sleep_interval=self.sleep_interval)
            ]

            # Check if scraping time exceeded limit
            if time.time() - start_time > self.scrape_timeout:
                print(f"Scraping for {company} timed out, adding company name and skipping...")
                data_rows.append({"Company": company})
                continue

            data = {"Company": company}
            for i, result in enumerate(search_results):
                if i < self.link_columns:
                    data[f"Link{i+1}"] = result

            data_rows.append(data)
            time.sleep(1)

        scraped_df = pd.DataFrame(data_rows)

        # Clean and extract domain names from links
        for col in scraped_df.columns[1:]:
            scraped_df[col] = scraped_df[col].apply(self.clean_url)
            scraped_df[col] = scraped_df[col].apply(self.extract_domain)

        return scraped_df

    def clean_url(self, url):
        """
        Removes unnecessary parts from URLs like "https" and "wikipedia".

        Args:
            url (str): The URL to clean.

        Returns:
            str: The cleaned URL, potentially containing the domain name.
        """

        if isinstance(url, str) and "wikipedia" in url and "https" in url:
            parts = url.split("/")
            for part in reversed(parts):
                if part and not part.startswith("wikipedia") and not part.startswith("https"):
                    return part
        return url

    def extract_domain(self, link):
        """
        Extracts the domain name from a URL.

        Args:
            link (str): The URL to extract the domain from.

        Returns:
            str: The extracted domain name, or the original link if no domain is found.
        """

        parsed_url = urlparse(str(link))
        if parsed_url.scheme and parsed_url.netloc:
            domain_parts = parsed_url.netloc.split(".")
            if domain_parts:
                cleaned_domain = domain_parts[-2] if len(domain_parts) > 1 else domain_parts[-1]
                return cleaned_domain
        # Fallback if no domain is found
        if ".".join(str(link).split(".")[1:-1]):
            return ".".join(str(link).split(".")[1:-1])
        else:
            return link
