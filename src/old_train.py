
def train_model(prepared_df, save=False):
    """Train the prediction model"""
    print("Training model...")
    
    if prepared_df is None or prepared_df.empty:
        print("No data available for training")
        return False
    
    # Drop rows with NaN values
    prepared_df = prepared_df.dropna(subset=self.feature_columns + ['home_win'])
    
    if len(prepared_df) < 10:
        print("Not enough data for training after removing NaN values")
        return False
    
    # Split features and target
    X = prepared_df[self.feature_columns]
    y = prepared_df['home_win']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training with {len(X_train)} games, testing with {len(X_test)} games")
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier())
    ])
    
    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
    
    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    self.model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = self.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    if save:
        joblib.dump(self.model, 'nba_prediction_model.pkl')
        joblib.dump(self.feature_columns, 'feature_columns.pkl')
        joblib.dump(self.team_stats_avg, 'team_stats_avg.pkl')
    
        print("Model saved to nba_prediction_model.pkl")
    
    return True
