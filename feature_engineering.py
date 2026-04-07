import numpy as np
import pandas as pd

def engineer_stability_features(df):
    """
    Creates state variables W (Wealth/Stability) and D (Debt Pressure).
    Uses static 'monthly_income' for robustness against sparse transaction data.
    """
    print("Engineering stability features...")
    
    # 1. Financial Stability Score (W_proxy)
    # We use Net Cash Flow from transactions divided by STATIC monthly income.
    # This tells us: "What fraction of their reported income are they actually saving?"
    df['Savings_Rate'] = df['Avg_Net_Cash_Flow'] / (df['monthly_income'] + 1e-5)
    
    # Clip Savings Rate to avoid extreme outliers (e.g., -5 to 1)
    df['Savings_Rate'] = df['Savings_Rate'].clip(-5, 1)
    
    # 2. Debt Pressure (D_proxy)
    # Use outstanding_debt / monthly_income (Static Profile)
    if 'outstanding_debt' in df.columns and 'monthly_income' in df.columns:
        df['Debt_Pressure'] = df['outstanding_debt'] / (df['monthly_income'] + 1e-5)
    elif 'credit_utilization_ratio' in df.columns:
        df['Debt_Pressure'] = df['credit_utilization_ratio']
    else:
        df['Debt_Pressure'] = 0
        
    # 3. Clean Data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"Feature engineering complete. {len(df)} workers in dataset.")
    return df