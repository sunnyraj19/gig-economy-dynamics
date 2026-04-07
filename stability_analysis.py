import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def estimate_cluster_stability_robust(df, cluster_col='Archetype'):
    """
    Robust Stability Analysis using Static Income profiles.
    """
    print("Performing Robust Stability Analysis...")
    
    results = {}
    
    # Features: Debt Pressure (Static)
    # Target: Savings Rate (Dynamic Net Cash Flow / Static Income)
    features = ['Debt_Pressure']
    target = 'Savings_Rate'
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id].copy()
        
        # Drop rows where specific features are NaN
        cluster_data.dropna(subset=features + [target], inplace=True)
        
        # Filter for valid monthly income to ensure we aren't analyzing ghost accounts
        cluster_data = cluster_data[cluster_data['monthly_income'] > 100]
        
        if len(cluster_data) < 50:
            print(f"  Cluster {cluster_id}: Skipped (Only {len(cluster_data)} valid samples)")
            continue
            
        X = cluster_data[features].values
        y = cluster_data[target].values
        
        try:
            reg = LinearRegression().fit(X, y)
            
            debt_sensitivity = reg.coef_[0]
            baseline_savings = reg.intercept_
            r_squared = reg.score(X, y)
            
            # Interpretation:
            # Negative Sensitivity: As Debt increases, Savings Rate drops -> Unstable.
            if debt_sensitivity < -0.05:
                status = "Unstable (High Debt Sensitivity)"
            elif debt_sensitivity < 0:
                status = "Moderately Sensitive"
            else:
                status = "Resilient (or Survivorship Bias)"
                
            results[cluster_id] = {
                'Debt_Sensitivity': debt_sensitivity,
                'Baseline_Savings_Rate': baseline_savings,
                'R_Squared': r_squared,
                'Sample_Size': len(cluster_data),
                'Status': status
            }
            
        except Exception as e:
            print(f"  Error in Cluster {cluster_id}: {e}")
            continue
            
    return results