import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def identify_archetypes(df, features=['Financial_Stability_Score', 'Debt_Pressure', 'Avg_Monthly_Income', 'Std_Monthly_Income']):
    """
    Clusters workers into financial archetypes using K-Means.
    Uses the Elbow Method for speed instead of Silhouette Score.
    """
    print("Identifying financial archetypes...")
    
    # Ensure we only use rows with valid data for these features
    df_clean = df.dropna(subset=features).copy()
    
    # 1. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    
    # 2. Find Best K using Elbow Method (Inertia) - Much Faster
    print("Calculating optimal clusters (Elbow Method)...")
    inertias = []
    K_range = range(2, 8) # Check K from 2 to 7
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  K={k}, Inertia={kmeans.inertia_:.2f}")
        
    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.grid(True)
    plt.savefig('../results/figures/elbow_method.png')
    plt.show()
    
    # Choose K=4 as a standard starting point for "Archetypes" 
    # (Stable, Struggling, High-Income/High-Debt, Precarious)
    best_k = 4 
    print(f"Selected K={best_k} for analysis.")
    
    # 3. Apply the Model
    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_clean['Archetype'] = kmeans_best.fit_predict(X_scaled)
    
    print("Clustering complete.")
    return df_clean, kmeans_best, scaler