import joblib
import pandas as pd
import numpy as np

def print_feature_importances():
    try:
        # Load the model and feature columns
        model = joblib.load('nba_prediction_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        # Get feature importances from the model
        # For Pipeline objects, we need to access the classifier
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            importances = classifier.feature_importances_
        else:
            importances = model.feature_importances_
        
        # Create a DataFrame to display feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        })
        
        # Sort by importance in descending order
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Print all features and their importances
        print("\nFeature Importances:")
        print("=" * 50)
        print(f"{'Feature':<40} {'Importance':<10}")
        print("-" * 50)
        
        for index, row in feature_importance_df.iterrows():
            print(f"{row['Feature']:<40} {row['Importance']:.6f}")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"Total number of features: {len(feature_columns)}")
        print(f"Top 5 most important features:")
        for i in range(min(5, len(feature_importance_df))):
            feature = feature_importance_df.iloc[i]['Feature']
            importance = feature_importance_df.iloc[i]['Importance']
            print(f"  {i+1}. {feature}: {importance:.6f}")
        
        # Create a simple visualization of relative importances
        print("\nRelative Importance:")
        for i in range(min(10, len(feature_importance_df))):
            feature = feature_importance_df.iloc[i]['Feature']
            importance = feature_importance_df.iloc[i]['Importance']
            bar_length = int(importance * 100)
            print(f"{feature:<40} {'#' * bar_length} {importance:.4f}")
        
        return feature_importance_df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've trained and saved the model first.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    print_feature_importances()
