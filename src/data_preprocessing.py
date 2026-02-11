"""
Data preprocessing pipeline for customer churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class DataPreprocessor:
    """Handle all data preprocessing steps"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load raw data"""
        print(f"📥 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} rows")
        return df
    
    def clean_data(self, df):
        """Clean and fix data issues"""
        print("🧹 Cleaning data...")
        
        df_clean = df.copy()
        
        # Fix TotalCharges (has spaces instead of numbers for new customers)
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'], 
            errors='coerce'
        )
        
        # Fill missing TotalCharges with MonthlyCharges (new customers)
        mask = df_clean['TotalCharges'].isnull()
        df_clean.loc[mask, 'TotalCharges'] = df_clean.loc[mask, 'MonthlyCharges']
        
        print(f"✅ Cleaned {mask.sum()} missing values")
        
        return df_clean
    
    def encode_target(self, df):
        """Encode target variable (Churn: Yes/No → 1/0)"""
        print("🎯 Encoding target variable...")
        
        df_encoded = df.copy()
        df_encoded['Churn'] = (df_encoded['Churn'] == 'Yes').astype(int)
        
        print(f"✅ Churn distribution: {df_encoded['Churn'].value_counts().to_dict()}")
        
        return df_encoded
    
    def encode_categoricals(self, df, fit=True):
        """Encode categorical variables"""
        print("📝 Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        # Binary categorical columns (Yes/No)
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
        
        # Multi-class categorical columns
        categorical_cols = [
            'gender', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df_encoded[col] = self.label_encoders[col].transform(
                            df_encoded[col].astype(str)
                        )
        
        print(f"✅ Encoded {len(binary_cols) + len(categorical_cols)} categorical features")
        
        return df_encoded
    
    def prepare_features(self, df):
        """Prepare feature matrix X and target y"""
        print("🔧 Preparing features...")
        
        # Drop customerID (not useful for prediction)
        features_df = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        
        X = features_df.values
        y = df['Churn'].values
        
        feature_names = features_df.columns.tolist()
        
        print(f"✅ Created feature matrix: {X.shape}")
        print(f"   Features: {feature_names}")
        
        return X, y, feature_names
    
    def scale_features(self, X_train, X_test, fit=True):
        """Scale numerical features"""
        print("📏 Scaling features...")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        print("✅ Features scaled")
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor for later use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"✅ Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load saved preprocessor"""
        preprocessor_data = joblib.load(filepath)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        print(f"✅ Preprocessor loaded from {filepath}")


def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Encode target
    df = preprocessor.encode_target(df)
    
    # Encode categoricals
    df = preprocessor.encode_categoricals(df, fit=True)
    
    # Prepare features
    X, y, feature_names = preprocessor.prepare_features(df)
    
    # Split data
    print("🔪 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training: {len(X_train):,} | Testing: {len(X_test):,}")
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(
        X_train, X_test, fit=True
    )
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print("✅ Processed data saved to data/processed/")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n✅ PREPROCESSING COMPLETE!\n")
    
    # Summary
    print("📊 PREPROCESSING SUMMARY:")
    print("="*60)
    print(f"Original data: {len(df):,} customers")
    print(f"Features: {len(feature_names)}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Churn rate (train): {y_train.mean():.2%}")
    print(f"Churn rate (test): {y_test.mean():.2%}")
    print("="*60)


if __name__ == "__main__":
    main()