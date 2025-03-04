import os
import pandas as pd
import json
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# Get Supabase credentials
DB_URL = os.getenv('DB_URL')
DB_KEY = os.getenv('DB_KEY')

# Remove trailing slash from URL if present
if DB_URL and DB_URL.endswith('/'):
    DB_URL = DB_URL[:-1]

print(f"Connecting to Supabase at: {DB_URL}")

# Initialize Supabase client
supabase = create_client(DB_URL, DB_KEY)

def query_games_table(limit=10):
    """Query the games table and return results as DataFrame"""
    try:
        response = supabase.table('games').select('*').limit(limit).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            print(f"\nSample data from games table ({len(df)} rows):")
            print(df)
            
            # Get column information
            print(f"\nColumns in games table:")
            for column in df.columns:
                print(f"- {column}: {df[column].dtype}")
            
            return df
        else:
            print("\nNo data found in games table")
            return None
    except Exception as e:
        print(f"\nError querying games table: {str(e)}")
        return None

def query_gamestats_table(limit=10):
    """Query the gamestats table and return results as DataFrame"""
    try:
        response = supabase.table('gamestats').select('*').limit(limit).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            print(f"\nSample data from gamestats table ({len(df)} rows):")
            
            # Display only a subset of columns for readability
            if len(df.columns) > 10:
                key_stats = ['game_id', 'team', 'points', 'assists', 'rebounds', 'steals', 'blocks']
                available_cols = [col for col in key_stats if col in df.columns]
                print(df[available_cols])
                print(f"...and {len(df.columns) - len(available_cols)} more columns")
            else:
                print(df)
            
            # Get column information
            print(f"\nColumns in gamestats table ({len(df.columns)} total):")
            for column in sorted(df.columns):
                print(f"- {column}: {df[column].dtype}")
            
            return df
        else:
            print("\nNo data found in gamestats table")
            return None
    except Exception as e:
        print(f"\nError querying gamestats table: {str(e)}")
        return None

def get_game_with_stats(game_id):
    """Get a specific game with its stats for both teams"""
    try:
        # Get game info
        game_response = supabase.table('games').select('*').eq('game_id', game_id).execute()
        
        # Get game stats for both teams
        stats_response = supabase.table('gamestats').select('*').eq('game_id', game_id).execute()
        
        if game_response.data and stats_response.data:
            game_df = pd.DataFrame(game_response.data)
            stats_df = pd.DataFrame(stats_response.data)
            
            print(f"\nGame information for {game_id}:")
            print(game_df)
            
            print(f"\nTeam statistics for {game_id}:")
            
            # Get home and away teams
            home_team = game_df['home_team'].iloc[0] if 'home_team' in game_df.columns else None
            away_team = game_df['away_team'].iloc[0] if 'away_team' in game_df.columns else None
            
            # Filter stats for home and away teams
            if home_team and away_team and 'team' in stats_df.columns:
                home_stats = stats_df[stats_df['team'] == home_team]
                away_stats = stats_df[stats_df['team'] == away_team]
                
                # Display key stats for both teams
                key_stats = ['points', 'assists', 'reboundstotal', 'steals', 'blocks', 
                             'fieldgoalspercentage', 'threepointersmade', 'threepointerattempted']
                
                available_cols = ['team'] + [col for col in key_stats if col in stats_df.columns]
                
                print("\nKey statistics comparison:")
                comparison_df = pd.DataFrame()
                
                if not home_stats.empty and not away_stats.empty:
                    for col in available_cols:
                        if col in stats_df.columns:
                            if col == 'team':
                                comparison_df[col] = [home_team, away_team]
                            else:
                                home_val = home_stats[col].iloc[0] if not home_stats.empty else None
                                away_val = away_stats[col].iloc[0] if not away_stats.empty else None
                                comparison_df[col] = [home_val, away_val]
                    
                    print(comparison_df)
                else:
                    print("Could not find separate home/away team stats")
                    print(stats_df)
            else:
                print(stats_df)
            
            return {
                'game': game_df,
                'stats': stats_df
            }
        else:
            print(f"\nNo data found for game {game_id}")
            return None
    except Exception as e:
        print(f"\nError retrieving game {game_id}: {str(e)}")
        return None

def analyze_team_stats():
    """Analyze team statistics to understand performance metrics"""
    try:
        # Get all game stats
        stats_response = supabase.table('gamestats').select('*').execute()
        
        if stats_response.data:
            stats_df = pd.DataFrame(stats_response.data)
            
            # Get numerical columns
            numerical_columns = stats_df.select_dtypes(include=['number']).columns.tolist()
            
            print(f"\nAnalyzing team statistics across {len(stats_df)} records")
            
            # Check if points column exists
            if 'points' in stats_df.columns:
                print("\nPoints distribution:")
                print(stats_df['points'].describe())
                
                # Top scoring teams
                print("\nTop 5 team scoring performances:")
                top_scoring = stats_df.sort_values('points', ascending=False).head(5)
                print(top_scoring[['game_id', 'team', 'points']])
            
            # Check for key shooting stats
            shooting_cols = [col for col in stats_df.columns if 'percentage' in col.lower()]
            if shooting_cols:
                print("\nShooting percentages:")
                print(stats_df[shooting_cols].describe().T[['mean', 'min', 'max']])
            
            # Check for correlation between key stats if they exist
            key_stats = [col for col in numerical_columns if col in [
                'points', 'assists', 'reboundstotal', 'steals', 'blocks', 
                'fieldgoalsmade', 'threepointersmade', 'turnovers'
            ]]
            
            if len(key_stats) > 1:
                print("\nCorrelation between key stats:")
                corr_matrix = stats_df[key_stats].corr()
                print(corr_matrix)
            
            return stats_df
        else:
            print("\nNo game stats found")
            return None
    except Exception as e:
        print(f"\nError analyzing team stats: {str(e)}")
        return None

def main():
    """Main function to explore the database"""
    print("Querying database tables...")
    
    # Query games table
    games_df = query_games_table()
    
    # Query gamestats table
    stats_df = query_gamestats_table()
    
    # If we have games, get detailed info for one game
    if games_df is not None and not games_df.empty:
        sample_game_id = games_df['game_id'].iloc[0]
        print(f"\nGetting detailed information for game {sample_game_id}")
        game_with_stats = get_game_with_stats(sample_game_id)
    
    # Analyze team statistics
    print("\nAnalyzing team statistics...")
    analyze_team_stats()
    
    # Show available tables
    try:
        # Try querying expected tables based on your db.py file
        tables_to_check = ['games', 'gamestats']
        
        print("\nChecking available tables:")
        for table in tables_to_check:
            try:
                response = supabase.table(table).select('*').limit(1).execute()
                if response.data:
                    print(f"✓ Table exists and has data: {table}")
                else:
                    print(f"✓ Table exists but is empty: {table}")
            except Exception as e:
                print(f"✗ Table not found or error: {table} - {str(e)}")
    except Exception as e:
        print(f"Error checking tables: {str(e)}")

if __name__ == "__main__":
    main()
