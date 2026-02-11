"""
Make predictions on new customer data
"""

import numpy as np
import pandas as pd
import joblib
import os

class ChurnPredictor:
    """Class for making churn predictions on new customers"""
    
    def __init__(self, model_path='models/best_model.pkl', 
                 preprocessor_path='models/preprocessor.pkl'):
        """Initialize predictor with trained model and preprocessor"""
        
        print("📥 Loading model and preprocessor...")
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load preprocessor
        preprocessor_data = joblib.load(preprocessor_path)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        
        # Load feature names
        with open('data/processed/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        # Load metadata
        self.metadata = joblib.load('models/model_metadata.pkl')
        
        print(f"✅ Loaded: {self.metadata['model_name']}")
        print(f"   Expected features: {len(self.feature_names)}")
    
    def preprocess_single_customer(self, customer_data):
        """Preprocess a single customer's data"""
        
        # Create DataFrame
        df = pd.DataFrame([customer_data])
        
        # Remove customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Encode binary columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = (df[col] == 'Yes').astype(int)
        
        # Encode categorical columns
        categorical_cols = [
            'gender', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Ensure correct feature order
        X = df[self.feature_names].values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, customer_data):
        """
        Predict churn for a single customer
        
        Parameters:
        -----------
        customer_data : dict
            Dictionary with customer features
            
        Returns:
        --------
        prediction : str
            'Churn' or 'No Churn'
        probability : float
            Probability of churn (0-1)
        """
        
        # Preprocess
        X = self.preprocess_single_customer(customer_data)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        prediction_label = 'Churn' if prediction == 1 else 'No Churn'
        
        return prediction_label, probability
    
    def predict_batch(self, customers_df):
        """
        Predict churn for multiple customers
        
        Parameters:
        -----------
        customers_df : DataFrame
            DataFrame with customer features
            
        Returns:
        --------
        results_df : DataFrame
            DataFrame with predictions and probabilities
        """
        
        predictions = []
        probabilities = []
        
        for idx, row in customers_df.iterrows():
            customer_data = row.to_dict()
            pred, prob = self.predict(customer_data)
            predictions.append(pred)
            probabilities.append(prob)
        
        results_df = customers_df.copy()
        results_df['Churn_Prediction'] = predictions
        results_df['Churn_Probability'] = probabilities
        results_df['Risk_Level'] = results_df['Churn_Probability'].apply(
            lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
        )
        
        return results_df

def demo_predictions():
    """Demonstrate predictions on example customers"""
    
    print("\n" + "="*60)
    print("CHURN PREDICTION DEMO")
    print("="*60 + "\n")
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Example customers
    customers = [
        {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 85.0,
            'TotalCharges': 170.0
        },
        {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 36,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 65.0,
            'TotalCharges': 2340.0
        },
        {
            'gender': 'Male',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 8,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 95.0,
            'TotalCharges': 760.0
        }
    ]
    
    # Make predictions
    print("🔮 Predicting churn for 3 customers:\n")
    
    for i, customer in enumerate(customers, 1):
        prediction, probability = predictor.predict(customer)
        
        risk_level = 'HIGH' if probability >= 0.7 else ('MEDIUM' if probability >= 0.4 else 'LOW')
        risk_emoji = '🔴' if risk_level == 'HIGH' else ('🟡' if risk_level == 'MEDIUM' else '🟢')
        
        print(f"Customer {i}:")
        print(f"  Contract: {customer['Contract']}")
        print(f"  Tenure: {customer['tenure']} months")
        print(f"  Monthly Charges: ${customer['MonthlyCharges']:.2f}")
        print(f"  Internet: {customer['InternetService']}")
        print(f"  ")
        print(f"  {risk_emoji} Prediction: {prediction}")
        print(f"  Churn Probability: {probability:.2%}")
        print(f"  Risk Level: {risk_level}")
        
        if probability >= 0.5:
            print(f"  ⚠️  Recommendation: Immediate retention campaign")
        else:
            print(f"  ✅ Recommendation: Standard engagement")
        
        print()
    
    print("="*60)

def main():
    """Main prediction demo"""
    demo_predictions()

if __name__ == "__main__":
    main()