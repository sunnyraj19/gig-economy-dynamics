import pandas as pd
import os

def load_and_merge_data(raw_dir='data/raw'):
    """
    Loads gig worker profiles and transaction history, 
    then merges them into a unified dataset.
    """
    print("Loading data...")
    
    # Load Data
    try:
        df_workers = pd.read_csv(os.path.join(raw_dir, 'gig_workers.csv'))
        df_trans = pd.read_csv(os.path.join(raw_dir, 'transactions.csv'))
    except FileNotFoundError:
        print("Error: Could not find CSV files in data/raw/")
        return None

    print(f"Loaded {len(df_workers)} workers and {len(df_trans)} transactions.")
    
    # --- FIX: Auto-detect the transaction type column ---
    # We now include 'merchant_category' which is the correct name in your file
    possible_type_cols = ['merchant_category', 'type', 'category', 'transaction_type']
    type_col = None
    for col in possible_type_cols:
        if col in df_trans.columns:
            type_col = col
            break
    
    if type_col is None:
        print(f"Error: Could not find a transaction type column. Available columns: {df_trans.columns}")
        return None
        
    print(f"Using column '{type_col}' for transaction types.")
    # -----------------------------------------------

    # Preprocess Transactions
    df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])
    df_trans['Month'] = df_trans['timestamp'].dt.to_period('M')
    
    # Categorize Flows (Income vs Expense)
    def categorize_flow(transaction_type):
        # Convert to string to handle any potential non-string types
        t_type = str(transaction_type).strip()
        
        # In this dataset, 'Transfer' usually indicates income from platforms/peers
        if t_type in ['Transfer']:
            return 'Income'
        # Everything else is treated as an expense for this model
        elif t_type in ['EMI_Payment', 'Fuel', 'Utility', 'Grocery', 'Medical', 
                        'Restaurant', 'Entertainment', 'Shopping', 'ATM', 'Withdrawal', 'Expense']:
            return 'Expense'
        else:
            # If it's a category we don't recognize, we'll count it as 'Other' 
            # but for simplicity in this model, let's treat unknowns as Expenses 
            # unless they are explicitly Income.
            return 'Expense'

    # Apply the function to the detected column
    df_trans['Flow_Type'] = df_trans[type_col].apply(categorize_flow)
    
    # Aggregate Monthly Flows
    monthly_flows = df_trans.groupby(['worker_id', 'Month', 'Flow_Type'])['amount'].sum().unstack(fill_value=0)
    
    # Ensure columns exist even if some types are missing
    if 'Income' not in monthly_flows.columns: monthly_flows['Income'] = 0
    if 'Expense' not in monthly_flows.columns: monthly_flows['Expense'] = 0
    
    monthly_flows = monthly_flows.reset_index()
    monthly_flows['Net_Cash_Flow'] = monthly_flows['Income'] - monthly_flows['Expense']
    
    # Aggregate to Worker Level (Mean/Volatility)
    worker_dynamics = monthly_flows.groupby('worker_id').agg(
        Avg_Monthly_Income=('Income', 'mean'),
        Std_Monthly_Income=('Income', 'std'),
        Avg_Monthly_Expense=('Expense', 'mean'),
        Avg_Net_Cash_Flow=('Net_Cash_Flow', 'mean'),
        Num_Months_Active=('Month', 'count')
    ).reset_index()
    
    # Fill NaN std (if worker has only 1 month of data)
    worker_dynamics['Std_Monthly_Income'] = worker_dynamics['Std_Monthly_Income'].fillna(0)
    
    # Merge with Static Profile
    # Find the worker ID column name in df_workers (usually the first column)
    worker_id_col = df_workers.columns[0]
    df_workers.rename(columns={worker_id_col: 'worker_id'}, inplace=True)
    
    df_combined = pd.merge(df_workers, worker_dynamics, on='worker_id', how='left')
    
    # Fill missing dynamics for workers with no transactions
    df_combined['Avg_Monthly_Income'] = df_combined['Avg_Monthly_Income'].fillna(0)
    df_combined['Std_Monthly_Income'] = df_combined['Std_Monthly_Income'].fillna(0)
    df_combined['Avg_Net_Cash_Flow'] = df_combined['Avg_Net_Cash_Flow'].fillna(0)
    
    print("Data merging complete.")
    return df_combined