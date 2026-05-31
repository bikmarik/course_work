import pandas as pd
import requests
import os
from io import StringIO

class Synchornizer:
    def __init__(self):
        pass
    def sync_sp500(self):
        print("Synchronizing S&P 500 constituents from Wikipedia...")
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        df = df.drop_duplicates(subset=['CIK'], keep='first')
        
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/sp500_master.csv", index=False)
        
        # Return list of (ticker, cik) tuples
        ticker_cik_pairs = list(zip(df['Symbol'], df['CIK']))
        
        print(f"Successfully synced {len(ticker_cik_pairs)} unique company tickers to data/sp500_master.csv!")
        
        return ticker_cik_pairs