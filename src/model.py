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

def fetch_data(requested_stats=None,all=True):
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
    
    if all:
        print("Fetching all stats")
        select_string='*'
    elif requested_stats==None:
        print("No requested stats, defaulting to fetch all stats")
        select_string = '*' 
    else:
        select_string = 'game_id, team,'+ ','.join(requested_stats)
    
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
    games_df, stats_df = pd.DataFrame(all_games), pd.DataFrame(all_data)
    return games_df, stats_df

def prepare_features(games_df, stats_df, key_stats): # Convert game_date to datetime and sort
        
    games_df['game_id'] = games_df['game_id'].astype(int) 
    stats_df['game_id'] = stats_df['game_id'].astype(int)
     
    games_df.set_index('game_id', inplace=True)
    stats_df.set_index(['game_id', 'team'], inplace=True)
    stats_df = stats_df[key_stats] #Filter by requested stats

    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    games_df.sort_values('game_date', inplace=True)  
    #merged_df = stats_df.merge(games_df, on='game_id', how='left') 
    teams = games_df['home_team'].unique()   
    team_averages = {}
    
    
    for team in teams:
        team_row = {stat:0 for stat in key_stats}
        team_row['count'] = 1
        team_averages[team] = team_row
           
    prepared_list = []
     
    for game in games_df.itertuples():
        home_team = game.home_team
        away_team = game.away_team
        try:
            home_stats = stats_df.loc[(game.Index, home_team)] 
            away_stats = stats_df.loc[(game.Index, away_team)] 
        except Exception as e:
            print(f"Missing game stats for {game.Index}")
            continue 
        n, m = team_averages[home_team]['count'], team_averages[away_team]['count']
        home_win = 1 if home_stats['points'] > away_stats['points'] else 0
         
        feature_row = {
            'home_team':home_team, 
            'away_team':away_team,
            'home_win': home_win, 
            'game_date': game.game_date
        }
        for name,home_stat in home_stats.items():
            if name in ['team', 'game_id'] or not isinstance(home_stat, (int, float)):
                continue
            
            prev_home_stat = team_averages[home_team][name]
            rolling_home_stat = ((n-1)/n)*prev_home_stat + (1/n)*home_stat
            
            away_stat = away_stats[name]
            prev_away_stat= team_averages[away_team][name]
            rolling_away_stat = ((m-1)/m)*prev_away_stat + (1/m)*away_stat
             
            feature_row['home_'+name] = rolling_home_stat
            feature_row['away_'+name] = rolling_away_stat

            team_averages[home_team][name] = rolling_home_stat
            team_averages[away_team][name] = rolling_away_stat
        
        team_averages[home_team]['count'] = n+1 
        team_averages[away_team]['count'] = m+1 
        prepared_list.append(feature_row)
        
     
    
    print(f"Prepared features for: {len(prepared_list)} games")
    prepared_df = pd.DataFrame(prepared_list)
    
    return prepared_df 

def train_model(prepared_df):
    return 

def main():
    games_path = '../static/games.csv'
    stats_path = '../static/gamestats.csv'
    key_stats = [
        'points', 'assists', 'reboundstotal', 'steals', 'blocks', 
        'fieldgoalspercentage', 'threepointersmade', 'threepointersattempted',
        'turnovers', 'fieldgoalsmade', 'fieldgoalsattempted'
    ]

    if os.path.exists(games_path) and os.path.exists(stats_path):
        print("Fetching raw data from csv files")
        raw_game_data, raw_stat_data = pd.read_csv(games_path), pd.read_csv(stats_path) #Store all data in csv to reuse
    else:          
        raw_game_data, raw_stat_data = fetch_data()
        print("Storing raw data in csv file")
        raw_game_data.to_csv(games_path)
        raw_stat_data.to_csv(stats_path)
    
    prepared_features = prepare_features(raw_game_data, raw_stat_data,key_stats)
    prepared_features.to_csv('../static/features.csv')
    train_model(prepared_features) 


if __name__ == "__main__":
    main()
