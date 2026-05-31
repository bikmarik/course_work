import pandas as pd
import numpy as np
import src.dataGenius.calculate_data as calculate_data 
import os

class FinancialProcessor:
    def __init__(self, ticker, year, sic=0):
        self.engine = calculate_data.DataCalculator()
        self.ticker = ticker
        self.year = year
        self.path = f"data/{ticker}/{year}"
        
        # Sector Classification Engine (Strictly Numeric)
        self.sic = int(sic) if sic else 0
        self.sector_idx, self.sector_vector = self._determine_sector()

    def _determine_sector(self):
        """
        Determines the Macro-Sector based on SIC.
        Returns the integer index (0-8) and the One-Hot array.
        Indices:
        0: Agri, 1: Mining, 2: Construction, 3: Mfg/Standard, 
        4: Trans/Util, 5: Wholesale, 6: Retail, 7: Finance, 8: Services
        """
        one_hot = [0.0] * 9 
        
        if self.sic == 0:
            idx = 3 # Default to 3 (Manufacturing / Standard Corp)
        else:
            first_two = self.sic // 100
            if 1 <= first_two <= 9:     idx = 0
            elif 10 <= first_two <= 14: idx = 1
            elif 15 <= first_two <= 17: idx = 2
            elif 20 <= first_two <= 39: idx = 3
            elif 40 <= first_two <= 49: idx = 4
            elif 50 <= first_two <= 51: idx = 5
            elif 52 <= first_two <= 59: idx = 6
            elif 60 <= first_two <= 67: idx = 7
            elif 70 <= first_two <= 89: idx = 8
            else:                       idx = 3
            
        one_hot[idx] = 1.0
        return idx, one_hot

    def get_value(self, df, include_words, exclude_words=None):
        try:
            if 'concept' not in df.columns: return 0.0, 0.0
            exclude_words = exclude_words or []
            
            mask = pd.Series(False, index=df.index)
            for word in include_words:
                mask = mask | df['concept'].str.contains(word, case=False, na=False)
                
            for word in exclude_words:
                mask = mask & ~df['concept'].str.contains(word, case=False, na=False)
                
            if 'dimension' in df.columns:
                mask = mask & (df['dimension'].astype(str).str.lower() == 'false')
                
            matches = df[mask].copy()
            if matches.empty: return 0.0, 0.0
                
            matches['tag_len'] = matches['concept'].str.len()
            matches = matches.sort_values('tag_len')
            best_row = matches.iloc[[0]]
            
            date_cols = [c for c in df.columns if isinstance(c, str) and c.count('-') == 2 and len(c) == 10]
            if not date_cols: return 0.0, 0.0
                
            current_val = pd.to_numeric(best_row[date_cols[0]].iloc[0], errors='coerce')
            prev_val = 0.0
            if len(date_cols) > 1:
                prev_val = pd.to_numeric(best_row[date_cols[1]].iloc[0], errors='coerce')
                
            return (0.0 if np.isnan(current_val) else float(current_val), 
                    0.0 if np.isnan(prev_val) else float(prev_val))
                    
        except Exception as e:
            print(f"⚠️ get_value Error on {include_words}: {e}")
            return 0.0, 0.0

    def process_ticker(self):
        if not os.path.exists(self.path):
            return None

        try:
            bs_df = pd.read_csv(f"{self.path}/balance_sheet.csv")
            is_df = pd.read_csv(f"{self.path}/income_statement.csv")
            cf_df = pd.read_csv(f"{self.path}/cashflow.csv")

            # 1. ASSETS & LIABILITIES
            assets, _ = self.get_value(bs_df, ["us-gaap_Assets"], ["Current", "Noncurrent", "Abstract"])
            liab_total, _ = self.get_value(bs_df, ["us-gaap_Liabilities"], ["Current", "Noncurrent", "Abstract"])
            
            if liab_total == 0:
                equity, _ = self.get_value(bs_df, ["StockholdersEquity", "PartnersCapital"], ["Abstract", "Including"])
                liab_total = assets - equity

            # ==========================================
            # 2. NUMERIC SECTOR ROUTING
            # ==========================================
            rev_excludes = ["Deferred", "Unearned", "Cost", "Obligation", "Future", "Abstract", "NetIncome", "OperatingIncome"]
            
            cogs, inv, rd = 0.0, 0.0, 0.0

            if self.sector_idx == 7: # Index 7 = Finance
                rev_curr, rev_prev = self.get_value(is_df, ["RevenuesNetOfInterestExpense", "TotalRevenuesAndOtherIncome", "InterestAndDividendIncome", "InterestIncome"], rev_excludes)
                cogs = 0.0
                inv = 0.0 
                
            elif self.sector_idx == 8: # Index 8 = Services
                rev_curr, rev_prev = self.get_value(is_df, ["Revenue", "ServicesRevenue"], rev_excludes)
                cogs = self.get_value(is_df, ["CostOfServices", "LaborAndRelatedExpense", "CostOfRevenue"], ["Depreciation", "Abstract"])[0]
                inv = 0.0 
                
            else: # Indices 0-6 = Standard / Mfg / Retail
                rev_curr, rev_prev = self.get_value(is_df, ["Revenue", "Sales", "NetSales"], rev_excludes)
                cogs = self.get_value(is_df, ["CostOfRevenue", "CostOfGoods", "CostOfSales"], ["Depreciation", "Abstract"])[0]
                inv = self.get_value(bs_df, ["InventoryNet", "Inventories"], ["Abstract"])[0]

            # ==========================================

            # 3. STANDARD INCOME / CASH FLOWS
            op_inc, _ = self.get_value(is_df, ["OperatingIncome"], ["Abstract"])
            net_inc, _ = self.get_value(is_df, ["NetIncomeLoss"], ["Abstract", "Noncontrolling", "Available"])
            cf_ops, _ = self.get_value(cf_df, ["NetCashProvidedByUsedInOperatingActivities"], ["Abstract"])
            depr_val, _ = self.get_value(cf_df, ["Depreciation"], ["Abstract"])
            rd = self.get_value(is_df, ["Research"], ["Acquired", "InProcess", "Abstract"])[0]
            sga = self.get_value(is_df, ["SellingGeneralAndAdministrative"], ["Abstract"])[0]
            capex = abs(self.get_value(cf_df, ["PaymentsToAcquirePropertyPlantAndEquipment"], ["Abstract"])[0])

            raw_data = {
                "Revenue": rev_curr if rev_curr != 0 else 1.0,
                "PrevRevenue": rev_prev if rev_prev != 0 else 1.0,
                "OperatingIncome": op_inc,
                "Depreciation": depr_val, 
                "NetIncome": net_inc,
                "Assets": assets if assets != 0 else 1.0,
                "Liabilities": liab_total if liab_total != 0 else 1.0,
                "MarketCap": assets - liab_total,
                "COGS": cogs,
                "SGA": sga,
                "RD": rd,
                "Inventory": inv,
                "CAPEX": capex,
                "AssetsCurrent": self.get_value(bs_df, ["AssetsCurrent"], ["Abstract"])[0],
                "LiabilitiesCurrent": self.get_value(bs_df, ["LiabilitiesCurrent"], ["Abstract"])[0],
                "RetainedEarnings": self.get_value(bs_df, ["RetainedEarnings"], ["Abstract"])[0],
                "CashFlowOps": cf_ops
            }

            # 4. FINAL TENSOR CONSTRUCTION
            base_tensor = self.engine.get_tensor(raw_data)
            
            if isinstance(base_tensor, np.ndarray):
                base_tensor = base_tensor.flatten().tolist()
                
            final_tensor = base_tensor + self.sector_vector

            return {
                "ticker": self.ticker,
                "sector_idx": self.sector_idx, # Kept purely numeric
                "tensor": final_tensor, 
                "raw_data": raw_data
            }

        except Exception as e:
            print(f"❌ Critical Pipeline Failure for {self.ticker}: {e}")
            return None