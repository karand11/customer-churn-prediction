"""
Train multiple models and compare performance
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_processed_data():
    """Load preprocessed data"""
    print("📥 Loading processed data...")
    
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✅ Data loaded: Train={len(X_train):,}, Test={len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, feature_names

def get_models():
    """Initialize models to compare"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    return models

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train all models and evaluate"""
    results = []
    trained_models = {}
    
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60 + "\n")
    
    for name, model in models.items():
        print(f"🤖 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        trained_models[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")
    
    results_df = pd.DataFrame(results)
    
    return results_df, trained_models

def plot_model_comparison(results_df):
    """Visualize model comparison"""
    print("📊 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for idx, metric in enumerate(metrics):
        axes[idx].barh(results_df['Model'], results_df[metric], color='steelblue', edgecolor='black')
        axes[idx].set_xlabel(metric, fontweight='bold')
        axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
        axes[idx].set_xlim(0, 1)
        axes[idx].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(results_df[metric]):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/model_comparison.png")
    plt.close()

def plot_confusion_matrices(trained_models, y_test):
    """Plot confusion matrices for all models"""
    print("📊 Creating confusion matrices...")
    
    n_models = len(trained_models)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (name, data) in enumerate(trained_models.items()):
        cm = confusion_matrix(y_test, data['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide last subplot if odd number
    if n_models < 6:
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/confusion_matrices.png")
    plt.close()

def plot_roc_curves(trained_models, y_test):
    """Plot ROC curves for all models"""
    print("📊 Creating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    for name, data in trained_models.items():
        fpr, tpr, _ = roc_curve(y_test, data['probabilities'])
        auc = roc_auc_score(y_test, data['probabilities'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/roc_curves.png")
    plt.close()

def save_best_model(results_df, trained_models):
    """Save the best performing model"""
    print("\n💾 Saving best model...")
    
    # Find best model by F1-score (balanced metric)
    best_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]['model']
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'metrics': results_df.loc[best_idx].to_dict()
    }
    joblib.dump(metadata, 'models/model_metadata.pkl')
    
    print(f"✅ Best model: {best_model_name}")
    print(f"   F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")
    print(f"   Saved to models/best_model.pkl")
    
    return best_model_name

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    
    # Get models
    models = get_models()
    
    # Train and evaluate
    results_df, trained_models = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )
    
    # Display results
    print("\n📊 MODEL COMPARISON RESULTS:")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_results.csv', index=False)
    print("\n✅ Results saved to results/model_results.csv")
    
    # Create visualizations
    plot_model_comparison(results_df)
    plot_confusion_matrices(trained_models, y_test)
    plot_roc_curves(trained_models, y_test)
    
    # Save best model
    best_model_name = save_best_model(results_df, trained_models)
    
    print("\n✅ TRAINING COMPLETE!\n")

if __name__ == "__main__":
    main()