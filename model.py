import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from db import query_games_table
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Get Supabase credentials
URL = os.getenv('DB_URL')
KEY = os.getenv('DB_KEY')

# Remove trailing slash from URL if present
"""
if DB_URL and DB_URL.endswith('/'):
    DB_URL = DB_URL[:-1]
"""
# Initialize Supabase client
supabase = create_client(URL, KEY)

class NBAGamePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.team_stats_avg = None
        
    def fetch_data(self, key_stats):
        def add(arr1, arr2):
            for a in arr1:
                arr2.append(a) 
            
        """Fetch games and stats data from Supabase"""
        print("Fetching data from Supabase...") 
        page_size = 1000 

        # Get all games
        start = page_size
        games_response = supabase.table('games').select('*').range(0,page_size).execute()
        games = games_response.data
        all_games = [] 
        add(games, all_games)
        while(len(games)==page_size):
            games_response = supabase.table('games').select('*').range(start,start+page_size).execute()
            games = games_response.data
            add(games, all_games)   
            start = start + page_size 
        

        # Get all game stats
        stat_string = ','.join(key_stats)
        select_string = 'game_id, team,' + stat_string
        all_data = []
        stats_response = supabase.table('gamestats').select(select_string).range(0,page_size).execute() 
        data = stats_response.data
        all_data.extend(data)
        start = page_size
        while(len(data)==page_size):
            stats_response = supabase.table('gamestats').select(select_string).range(start,start+page_size).execute() 
            data = stats_response.data
            all_data.extend(data)
            start = start+page_size 
        #return data frames  
        games_df, stats_df = pd.DataFrame(all_games).set_index('game_id'), pd.DataFrame(all_data).set_index(['game_id','team'])
        print(f"Fetched {len(games_df)} games and {len(stats_df)} team stat records")
        return games_df, stats_df
    def prepare_features2(self, games_df, stats_df, key_stats):
        
        games_df['game_date'] = pd.to_datetime(games_df['game_date'])
        games_df.sort_values('game_date', inplace=True)  
        #merged_df = stats_df.merge(games_df, on='game_id', how='left') 
        teams = games_df['home_team'].unique()   
        team_averages = []
        
        for team in teams:
            team_row = {}
            team_row['team'] = team
            team_row['count'] = 1
            for stat in list(stats_df.columns):
                team_row[stat] = 0
            team_averages.append(team_row)
        
        team_averages_df = pd.DataFrame(team_averages)
        prepared_list = []
        
        for game in games_df.itertuples(index=False):
            home_team = game.home_team
            away_team = game.away_team
            home_stats = stats_df.loc[(game.game_id, home_team)] if (game.game_id, home_team) in stats_df.index else None
            away_stats = stats_df.loc[(game.game_id, away_team)] if (game.game_id, away_team) in stats_df.index else None
            
            if home_stats is None or away_stats is None:
                print("missing stats")
                continue  
            n, m = team_averages_df.loc[team_averages_df['team'] == home_team, 'count'].values[0], team_averages_df.loc[team_averages_df['team'] == away_team, 'count'].values[0]
            
            home_win = 1 if home_stats['points'] > away_stats['points'] else 0
             
            feature_row = {
                'home_team':home_team, 
                'away_team':away_team,
                'home_win': home_win, 
                'game_date': game['game_date']
            }
            for name,stat in home_stats.items():
                if name == 'team' or name == 'game_id' or not isinstance(stat, (int, float)):
                    continue
                prev_home_stat = team_averages_df[team_averages_df['team']==home_team]
                rolling_home_stat = ((n-1)/n)*prev_home_stat[name] + (1/n)*stat
                prev_away_stat= team_averages_df[team_averages_df['team']==away_team]
                rolling_away_stat = ((m-1)/m)*prev_away_stat[name] + (1/m)*stat
                 
                feature_row['home_'+name] = rolling_home_stat
                feature_row['away_'+name] = rolling_away_stat

                team_averages_df.loc[team_averages_df['team'] == home_team, name] = rolling_home_stat
                team_averages_df.loc[team_averages_df['team'] == away_team, name] = rolling_away_stat
            
            team_averages_df.loc[team_averages_df['team'] == home_team, 'count'] = n+1 
            team_averages_df.loc[team_averages_df['team'] == away_team, 'count'] = m+1 
            prepared_list.append(feature_row)
            print("hello")         
        print(len(prepared_list))
        return
    def prepare_features(self, games_df, stats_df, key_stats):
        """Prepare features for the model"""
        print("Preparing features...")
        print(key_stats) 
        # Ensure game_id is string type in both dataframes
        games_df['game_id'] = games_df['game_id'].astype(str)
        stats_df['game_id'] = stats_df['game_id'].astype(str)
        
        # Convert game_date to datetime
        games_df['game_date'] = pd.to_datetime(games_df['game_date'])
        
        # Sort games by date
        games_df = games_df.sort_values('game_date')
        
        # Create a list to store prepared data
        prepared_data = []
        
        
        # Filter to include only stats columns that exist in the dataframe
        available_stats = [col for col in key_stats if col in stats_df.columns]
        
        if not available_stats:
            print("Warning: None of the expected stats columns were found!")
            print(f"Available columns: {stats_df.columns.tolist()}")
            return None
        
        print(f"Using these statistics as features: {available_stats}")
        
        # Calculate rolling averages for each team
        teams = set(games_df['home_team'].unique()) | set(games_df['away_team'].unique())
        team_stats_history = {team: pd.DataFrame() for team in teams}
        
        # Process each game
        for _, game in games_df.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Get stats for this game
            game_stats = stats_df[stats_df['game_id'] == game_id]
            
            if len(game_stats) < 2:
                print(f"Warning: Missing stats for game {game_id} and {game_stats['points']}")
                continue
            
            # Get home and away stats
            home_stats = game_stats[game_stats['team'] == home_team]
            away_stats = game_stats[game_stats['team'] == away_team]
            
            if home_stats.empty or away_stats.empty:
                print(f"Warning: Missing team stats for game {game_id}")
                continue
            
            # Calculate team averages up to this point (excluding this game)
            home_avg = team_stats_history[home_team][available_stats].mean() if not team_stats_history[home_team].empty else pd.Series(0, index=available_stats)
            away_avg = team_stats_history[away_team][available_stats].mean() if not team_stats_history[away_team].empty else pd.Series(0, index=available_stats)
            
            # Determine the winner (1 for home team, 0 for away team)
            if 'points' in home_stats.columns and 'points' in away_stats.columns:
                home_points = home_stats['points'].iloc[0]
                away_points = away_stats['points'].iloc[0]
                home_win = 1 if home_points > away_points else 0
            else:
                # If points aren't available, skip this game
                continue
            
            # Create feature row
            feature_row = {
                'game_id': game_id,
                'game_date': game['game_date'],
                'home_team': home_team,
                'away_team': away_team,
                'home_win': home_win
            }
            
            # Add team average stats as features
            for stat in available_stats:
                if not home_avg.empty and stat in home_avg:
                    feature_row[f'home_avg_{stat}'] = home_avg[stat]
                else:
                    feature_row[f'home_avg_{stat}'] = 0
                    
                if not away_avg.empty and stat in away_avg:
                    feature_row[f'away_avg_{stat}'] = away_avg[stat]
                else:
                    feature_row[f'away_avg_{stat}'] = 0
            
            # Add this game's stats to team history
            if not home_stats.empty:
                team_stats_history[home_team] = pd.concat([team_stats_history[home_team], home_stats[available_stats]], ignore_index=True)
            
            if not away_stats.empty:
                team_stats_history[away_team] = pd.concat([team_stats_history[away_team], away_stats[available_stats]], ignore_index=True)
            
            prepared_data.append(feature_row)
        
        # Convert to DataFrame
        prepared_df = pd.DataFrame(prepared_data)
        
        # Store team average stats for future predictions
        self.team_stats_avg = {team: df[available_stats].mean() for team, df in team_stats_history.items()}
        
        # Store feature columns
        self.feature_columns = [col for col in prepared_df.columns if col.startswith('home_avg_') or col.startswith('away_avg_')]
        
        print(f"Prepared {len(prepared_df)} games with {len(self.feature_columns)} features")
                
        return prepared_df
    
    def train_model(self, prepared_df, save=False):
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
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            self.model = joblib.load('nba_prediction_model.pkl')
            self.feature_columns = joblib.load('feature_columns.pkl')
            self.team_stats_avg = joblib.load('team_stats_avg.pkl')
            print("Model loaded successfully")
            return True
        except FileNotFoundError:
            print("No saved model found")
            return False
    
    def predict_game(self, home_team, away_team):
        """Predict the outcome of a game between two teams"""
        if self.model is None:
            print("Model not trained or loaded")
            return None
        
        if home_team not in self.team_stats_avg or away_team not in self.team_stats_avg:
            print(f"Stats not available for {home_team} or {away_team}")
            return None
        
        # Create feature vector
        features = {}
        
        for stat in self.team_stats_avg[home_team].index:
            features[f'home_avg_{stat}'] = self.team_stats_avg[home_team][stat]
            features[f'away_avg_{stat}'] = self.team_stats_avg[away_team][stat]
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Select only the features used by the model
        features_df = features_df[self.feature_columns]
        
        # Make prediction
        win_prob = self.model.predict_proba(features_df)[0, 1]
        prediction = "Home team win" if win_prob > 0.5 else "Away team win"
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': win_prob,
            'prediction': prediction
        }
    
    def predict_upcoming_games(self, days=7):
        """Predict outcomes for upcoming games"""
        # Fetch upcoming games from the database
        today = datetime.now()
        end_date = today + timedelta(days=days)
        
        # Format dates for the query
        today_str = today.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Query upcoming games
        upcoming_response = supabase.table('games').select('*').gte('game_date', today_str).lte('game_date', end_date_str).execute()
        
        if not upcoming_response.data:
            print(f"No upcoming games found between {today_str} and {end_date_str}")
            return []
        
        upcoming_games = pd.DataFrame(upcoming_response.data)
        
        predictions = []
        for _, game in upcoming_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = game['game_date']
            
            prediction = self.predict_game(home_team, away_team)
            if prediction:
                prediction['game_id'] = game['game_id']
                prediction['game_date'] = game_date
                predictions.append(prediction)
        
        return predictions

def main():
    # Create predictor
    args = sys.argv
    save_val = False
    if len(args) == 2:
        if args[1] == 1:
            save_val= True
            
    predictor = NBAGamePredictor()
         # Define key statistics to use as features
    key_stats = [
        'points', 'assists', 'reboundstotal', 'steals', 'blocks', 
        'fieldgoalspercentage', 'threepointersmade', 'threepointersattempted',
        'turnovers', 'fieldgoalsmade', 'fieldgoalsattempted'
    ]

    # Try to load existing model
    if not predictor.load_model():
        # If no model exists, fetch data and train a new one
        games_df, stats_df = predictor.fetch_data(key_stats)
        prepared_df = predictor.prepare_features2(games_df, stats_df, key_stats) 
        predictor.train_model(prepared_df,save_val)
    
    # Predict upcoming games
    print("\nPredicting upcoming games:")
    predictions = predictor.predict_upcoming_games(days=14)
    
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        print(predictions_df[['game_date', 'home_team', 'away_team', 'home_win_probability', 'prediction']])
    
    # Example of predicting a specific matchup
    print("\nExample prediction for a specific matchup:")
    prediction = predictor.predict_game('BOS', 'LAL')
    if prediction:
        print(f"Boston Celtics vs Los Angeles Lakers")
        print(f"Home win probability: {prediction['home_win_probability']:.4f}")
        print(f"Prediction: {prediction['prediction']}")

if __name__ == "__main__":
    main()
