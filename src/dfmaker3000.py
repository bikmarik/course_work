import numpy as np
from src.dataGenius.process_data import FinancialProcessor

class DataFrameBuilder3000:
    def __init__(self, ticker_cik_pairs, start_year, end_year):
        self.ticker_cik_pairs = ticker_cik_pairs
        self.start_year = start_year
        self.end_year = end_year

    def process_historical_ticker(self, ticker):
        history = []
        for year in range(self.start_year, self.end_year + 1):
            try:
                processor = FinancialProcessor(ticker, year)
                #print(f"Processing {ticker} {year}...")
                result = processor.process_ticker()
                if result and result['tensor'] is not None:
                    history.append(result['tensor'])
            except Exception as e:
                print(f"Failed on {ticker} {year}: {e}")
        return np.array(history)

    def build_seq2seq_dataset(self, input_window=3, output_window=1):
        """
        Converts raw history into [Input: N Years] -> [Target: M Years]
        Wall Street Standard for Annual Data: 3 -> 1
        """
        X, Y = [], []
        
        # Calculate the total span of years needed for one complete sample
        total_window = input_window + output_window
        
        for ticker, cik in self.ticker_cik_pairs:
            history = self.process_historical_ticker(ticker)
            
            # If we need 4 years total (3 in, 1 out) and only have 3, skip it.
            if len(history) < total_window:
                continue
                
            # Sliding window across the company's timeline
            for i in range(len(history) - total_window + 1):
                # Grab the exact chunk for inputs
                input_seq = history[i : i + input_window]
                
                # Grab the exact chunk for the future targets
                target_seq = history[i + input_window : i + total_window]
                
                X.append(input_seq)
                Y.append(target_seq)
        
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        print(f"Final arrays - X shape: {X_arr.shape}, Y shape: {Y_arr.shape}")
        
        return X_arr, Y_arr