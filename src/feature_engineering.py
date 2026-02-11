"""
Advanced feature engineering for churn prediction
"""

import pandas as pd
import numpy as np

def create_advanced_features(df):
    """
    Create additional features to improve model performance
    
    Parameters:
    -----------
    df : DataFrame
        Original dataframe with basic features
        
    Returns:
    --------
    df_enhanced : DataFrame
        DataFrame with additional engineered features
    """
    
    print("🔧 Engineering advanced features...")
    
    df_enhanced = df.copy()
    
    # 1. Tenure-based features
    df_enhanced['TenureGroup'] = pd.cut(
        df_enhanced['tenure'],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
    )
    
    # 2. Charge-based features
    df_enhanced['AvgMonthlyCharges'] = df_enhanced['TotalCharges'] / (df_enhanced['tenure'] + 1)
    df_enhanced['ChargePerService'] = df_enhanced['MonthlyCharges'] / (
        df_enhanced[['PhoneService', 'InternetService', 'OnlineSecurity', 
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                     'StreamingTV', 'StreamingMovies']].apply(
            lambda x: (x != 'No').sum(), axis=1
        ) + 1
    )
    
    # 3. Service count features
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies']
    
    df_enhanced['TotalServices'] = df_enhanced[service_cols].apply(
        lambda x: (x != 'No').sum(), axis=1
    )
    
    # 4. Customer value segments
    df_enhanced['CustomerValue'] = pd.cut(
        df_enhanced['TotalCharges'],
        bins=[0, 500, 2000, 5000, np.inf],
        labels=['Low', 'Medium', 'High', 'Premium']
    )
    
    # 5. Risk indicators
    df_enhanced['HasTechSupport'] = (df_enhanced['TechSupport'] == 'Yes').astype(int)
    df_enhanced['HasOnlineSecurity'] = (df_enhanced['OnlineSecurity'] == 'Yes').astype(int)
    df_enhanced['IsMonthToMonth'] = (df_enhanced['Contract'] == 'Month-to-month').astype(int)
    df_enhanced['IsFiberOptic'] = (df_enhanced['InternetService'] == 'Fiber optic').astype(int)
    
    print(f"✅ Created {df_enhanced.shape[1] - df.shape[1]} new features")
    
    return df_enhanced

if __name__ == "__main__":
    # Example usage
    print("Feature engineering module")
    print("Import this module to use create_advanced_features()")