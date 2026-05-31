import time
import random
import edgar
import os


class DataCollector:
    def __init__(self, ticker, cik, year):
        self.ticker = ticker
        self.cik = cik
        self.year = year
        self.data_path = f"data/{ticker}/{year}"
        if os.path.exists(self.data_path):
            print(f"  [+] Data for {ticker} {year} already exists. Skipping collection.")
            self.financials = None
            self.mda = None
            return
        os.makedirs(self.data_path, exist_ok=True)
        self.company = edgar.Company(cik)
        try:
            filing = self.company.get_filings(form="10-K", year=year, amendments=False)[0].obj()
        except IndexError:
            print(f"  [-] No 10-K filing found for {ticker} in {year}. Skipping...")
            self.financials = None
            self.mda = None
            return
        self.financials = filing.financials
        # Inside your DataCollector script
        try:
            # Attempt to grab the MDA
            self.mda = filing["Item 7"]
        except (AttributeError, KeyError, Exception) as e:
            print(f"⚠️ MDA Parsing failed for {self.ticker} {self.year}: {e}")
            self.mda = "" # Fallback to empty string so the rest of the script works

    def save_data(self):
        if not self.financials:
            print(f"  [-] No financials found for {self.ticker}")
            return False

        try:
            # SAVE BALANCE SHEET (QUANTIVESTA Constraints)
            bs_obj = self.financials.balance_sheet()
            if bs_obj:
                bs_obj.to_dataframe().to_csv(f"{self.data_path}/balance_sheet.csv")
                print(f"  [+] Saved Balance Sheet")

            # SAVE INCOME STATEMENT (PFAI Performance Drivers)
            is_obj = self.financials.income_statement()
            if is_obj:
                is_obj.to_dataframe().to_csv(f"{self.data_path}/income_statement.csv")
                print(f"  [+] Saved Income Statement")

            # SAVE CASH FLOW (The Reality Check)
            cf_obj = self.financials.cashflow_statement()
            if cf_obj:
                cf_obj.to_dataframe().to_csv(f"{self.data_path}/cashflow.csv")
                print(f"  [+] Saved Cash Flow")

            # SAVE ITEM 7 (MD&A Text for LLM processing)
            if self.mda:
                # Use management_discussion for direct access in 2026
                with open(f"{self.data_path}/mda.txt", "w", encoding="utf-8") as f:
                    f.write(str(self.mda))
                print(f"  [+] Saved MD&A Text")
            
            return True
        except Exception as e:
            print(f"  [-] Error saving {self.ticker}: {e}")
            return False