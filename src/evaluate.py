"""
Detailed evaluation of the best model
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve
)
import os

def load_model_and_data():
    """Load best model and test data"""
    print("📥 Loading model and data...")
    
    # Load model
    model = joblib.load('models/best_model.pkl')
    metadata = joblib.load('models/model_metadata.pkl')
    
    # Load test data
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Load feature names
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✅ Loaded model: {metadata['model_name']}")
    print(f"   Test samples: {len(X_test):,}")
    
    return model, X_test, y_test, feature_names, metadata

def detailed_classification_report(model, X_test, y_test):
    """Generate detailed classification report"""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60 + "\n")
    
    y_pred = model.predict(X_test)
    
    report = classification_report(
        y_test, y_pred,
        target_names=['No Churn', 'Churn'],
        digits=4
    )
    
    print(report)
    
    # Save report
    with open('results/classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print("✅ Report saved to results/classification_report.txt")

def plot_feature_importance(model, feature_names):
    """Plot feature importance (for tree-based models)"""
    print("\n📊 Analyzing feature importance...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Top 15 features
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[top_indices], color='teal', edgecolor='black', alpha=0.8)
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance Score', fontweight='bold', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✅ Saved to results/feature_importance.png")
        plt.close()
        
        # Save feature importance to CSV
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv('results/feature_importance.csv', index=False)
        print("✅ Feature importance saved to results/feature_importance.csv")
        
        # Print top features
        print("\n🔝 Top 10 Most Important Features:")
        print("="*60)
        for i in range(min(10, len(importance_df))):
            row = importance_df.iloc[i]
            print(f"{i+1:2d}. {row['Feature']:25s} {row['Importance']:.4f}")
        print("="*60)
    else:
        print("⚠️  Model doesn't have feature_importances_ attribute")

def plot_precision_recall_curve(model, X_test, y_test):
    """Plot precision-recall curve"""
    print("\n📊 Creating precision-recall curve...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2.5, color='darkblue', label='Precision-Recall Curve')
    plt.xlabel('Recall', fontweight='bold', fontsize=12)
    plt.ylabel('Precision', fontweight='bold', fontsize=12)
    plt.title('Precision-Recall Curve', fontweight='bold', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/precision_recall_curve.png")
    plt.close()

def analyze_errors(model, X_test, y_test, feature_names):
    """Analyze prediction errors"""
    print("\n🔍 Analyzing prediction errors...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # False positives and false negatives
    fp_mask = (y_test == 0) & (y_pred == 1)  # Predicted churn but didn't
    fn_mask = (y_test == 1) & (y_pred == 0)  # Missed churners
    
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    
    print(f"\n📊 Error Analysis:")
    print("="*60)
    print(f"False Positives: {fp_count} (predicted churn, but customer stayed)")
    print(f"False Negatives: {fn_count} (missed churners - CRITICAL)")
    print("="*60)
    
    # Analyze false negatives (most important to understand)
    if fn_count > 0:
        print(f"\n⚠️  Analyzing {fn_count} False Negatives (Missed Churners):")
        fn_probabilities = y_pred_proba[fn_mask]
        print(f"   Average churn probability: {fn_probabilities.mean():.3f}")
        print(f"   Min churn probability: {fn_probabilities.min():.3f}")
        print(f"   Max churn probability: {fn_probabilities.max():.3f}")
        
        # These are customers who churned but model didn't catch
        print("\n   💡 Insight: These churners had low predicted probabilities")
        print("      Consider lowering the decision threshold or adding more features")

def plot_probability_distribution(model, X_test, y_test):
    """Plot distribution of predicted probabilities"""
    print("\n📊 Creating probability distribution plot...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(10, 6))
    
    # Histogram for churned vs not churned
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, 
             label='No Churn (Actual)', color='green', edgecolor='black')
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, 
             label='Churn (Actual)', color='red', edgecolor='black')
    
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Predicted Churn Probability', fontweight='bold', fontsize=12)
    plt.ylabel('Frequency', fontweight='bold', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontweight='bold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('results/probability_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/probability_distribution.png")
    plt.close()

def business_insights(model, X_test, y_test):
    """Generate business insights"""
    print("\n💼 BUSINESS INSIGHTS")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Segment customers by risk
    high_risk = (y_pred_proba >= 0.7).sum()
    medium_risk = ((y_pred_proba >= 0.4) & (y_pred_proba < 0.7)).sum()
    low_risk = (y_pred_proba < 0.4).sum()
    
    print(f"\n🎯 Customer Risk Segmentation:")
    print(f"   High Risk (≥70% churn prob): {high_risk} customers ({high_risk/len(y_test)*100:.1f}%)")
    print(f"   Medium Risk (40-70%): {medium_risk} customers ({medium_risk/len(y_test)*100:.1f}%)")
    print(f"   Low Risk (<40%): {low_risk} customers ({low_risk/len(y_test)*100:.1f}%)")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Focus retention efforts on {high_risk} high-risk customers")
    print(f"   2. Monitor {medium_risk} medium-risk customers closely")
    print(f"   3. Maintain satisfaction for {low_risk} low-risk customers")
    
    # Calculate potential savings
    avg_customer_value = 1000  # Example: $1000 per customer per year
    retention_cost = 100  # Example: $100 to retain a customer
    
    true_churners = y_test.sum()
    detected_churners = ((y_test == 1) & (y_pred == 1)).sum()
    
    potential_revenue_saved = detected_churners * avg_customer_value
    retention_campaign_cost = (y_pred == 1).sum() * retention_cost
    net_value = potential_revenue_saved - retention_campaign_cost
    
    print(f"\n💰 Estimated Business Impact (example values):")
    print(f"   Total churners in test set: {true_churners}")
    print(f"   Churners detected by model: {detected_churners}")
    print(f"   Detection rate: {detected_churners/true_churners*100:.1f}%")
    print(f"   ")
    print(f"   Potential revenue saved: ${potential_revenue_saved:,}")
    print(f"   Retention campaign cost: ${retention_campaign_cost:,}")
    print(f"   Net value: ${net_value:,}")
    
    print("="*60)

def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("DETAILED MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model and data
    model, X_test, y_test, feature_names, metadata = load_model_and_data()
    
    # Display model info
    print(f"🤖 Model: {metadata['model_name']}")
    print(f"📅 Trained: {metadata.get('training_date', 'Unknown')}")
    print(f"📊 Training Metrics:")
    for metric, value in metadata['metrics'].items():
        if metric != 'Model':
            print(f"   {metric}: {value:.4f}")
    
    # Detailed classification report
    detailed_classification_report(model, X_test, y_test)
    
    # Feature importance
    plot_feature_importance(model, feature_names)
    
    # Additional plots
    plot_precision_recall_curve(model, X_test, y_test)
    plot_probability_distribution(model, X_test, y_test)
    
    # Error analysis
    analyze_errors(model, X_test, y_test, feature_names)
    
    # Business insights
    business_insights(model, X_test, y_test)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\n📁 All results saved to results/")
    print("   - classification_report.txt")
    print("   - feature_importance.png")
    print("   - feature_importance.csv")
    print("   - precision_recall_curve.png")
    print("   - probability_distribution.png\n")


if __name__ == "__main__":
    main()