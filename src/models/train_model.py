import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def load_data():
    """Load the preprocessed data."""
    processed_data_path = 'data/processed/processed_laptop_data.csv'
    return pd.read_csv(processed_data_path)

def prepare_features(df):
    """Prepare features for training."""
    # Drop non-numeric columns that we don't want to use for prediction
    columns_to_drop = ['name']  # Add any other non-numeric columns that shouldn't be used
    
    # Select only numeric columns for features
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    X = df[numeric_columns].drop(['price'], axis=1, errors='ignore')
    y = df['price']
    
    print("\nFeatures used for training:")
    print(X.columns.tolist())
    print(f"\nShape of feature matrix: {X.shape}")
    
    return X, y

def train_models(X, y):
    """Train and compare different models."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize models
    models = {
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        }
    }
    
    best_models = {}
    model_scores = {}
    
    # Train and evaluate each model
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        # Perform GridSearch
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        best_models[name] = grid_search.best_estimator_
        
        # Make predictions
        y_pred = grid_search.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        model_scores[name] = {
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
            'Best Parameters': grid_search.best_params_
        }
        
        print(f"{name} Results:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Best Parameters: {grid_search.best_params_}")
    
    # Select best model based on R2 score
    best_model_name = max(model_scores.items(), key=lambda x: x[1]['R2'])[0]
    best_model = best_models[best_model_name]
    
    return best_model, best_model_name, model_scores

def save_model(model, model_name):
    """Save the trained model."""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = f'models/{model_name}_model.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    """Main function to run the training pipeline."""
    print("Loading data...")
    df = load_data()
    
    print("Preparing features...")
    X, y = prepare_features(df)
    
    print("Training models...")
    best_model, best_model_name, model_scores = train_models(X, y)
    
    print(f"\nBest performing model: {best_model_name}")
    save_model(best_model, best_model_name)
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.joblib')

if __name__ == "__main__":
    main() 