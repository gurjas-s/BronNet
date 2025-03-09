import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv, dotenv_values
import requests
import pandas as pd

SEASONS = {"REGULAR":2, "PLAYOFFS":5}
MAX_GAMES = 1230
TABLES = {1:"games", 2:"gamestats"}


    
        
        
"""
The NBA's Game ID, 0021400001, is a 10-digit code: XXXYYGGGGG, where XXX refers to a season prefix, YY is the season year (e.g. 14 for 2014-15), and GGGGG refers to the game number (1-1230 for a full 30-team regular season).
"""

        
def upsert(client, tableName, rows):
    try:
        
        response = (

            client.table(tableName).upsert(rows).execute()
        )
        print("Added games to DB")
        return response
    except Exception as e:
        
        print(f"Error during upsert for {tableName}: {e}")
        raise
 
def getAllGames(client, seasonType, year): #Adds/updates all games from the year and season type to the database
    gameList = [] #Stores all game Id's for the given season type and year
    gameRows = []
    statRows = []
    start = 700
    batchSize = 100
    for i in range(1,MAX_GAMES):
        gameId = f"00{seasonType}{year}{i:05d}"
        gameList.append(gameId)
    
    for i in range(start,len(gameList)):
        
        if len(gameRows)>=batchSize:
            try:
                upsert(client, TABLES[1], gameRows) 
                upsert(client, TABLES[2], statRows)
                gameRows = []
                statRows = []
                print("Added 100 games")
            except Exception as error:
                print(f"Error: {error}")
                gameRows = []
                statRows = []

        url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameList[i]}.json"
        response = requests.get(url)
        if response.status_code != 200:
            print("Game not played yet")
            continue
        data = response.json()


        time = data["game"]["gameTimeLocal"] 
        home = data["game"]["homeTeam"]["teamTricode"] 
        away = data["game"]["awayTeam"]["teamTricode"]
        
        statKeys = data["game"]["homeTeam"]["statistics"].keys()
        homeStatRow = {"game_id": gameList[i], "team": home}
                
        awayStatRow = {"game_id": gameList[i], "team": away}
                

        for key in statKeys:
            if key.lower() == "steamfieldgoalattempts": #bug in nba games 100-200
                homeStatRow["teamfieldgoalattempts"] = data["game"]["homeTeam"]["statistics"].get(key,0)
                awayStatRow["teamfieldgoalattempts"] = data["game"]["awayTeam"]["statistics"].get(key,0)
            else:
                homeStatRow[key.lower()] = data["game"]["homeTeam"]["statistics"].get(key,0)
                awayStatRow[key.lower()] = data["game"]["awayTeam"]["statistics"].get(key,0)
        #print(homeStatRow) 
        
        gameRow = {"game_id": gameList[i], "home_team": home, "away_team":away, "game_date":time}
        
        gameRows.append(gameRow)
        statRows.append(homeStatRow)
        statRows.append(awayStatRow)
        print(f"Added game {i} to arrays")
    
    try:
        upsert(client, TABLES[1], gameRows) 
        upsert(client,TABLES[2], statRows)
        gameRows = []
        statRows = []
        print("Added last amount of games")
    except Exception as error:
        print(f"Error: {error}")
        return 

def getTodayGames(client): #Adds/updates today game's to database
    try: 
        gameList = [] 
        gameRows = []
        statRows = []

        scoreboard = requests.get("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json").json()
        gamesToday = scoreboard["scoreboard"]["games"]
        for game in gamesToday:
            gameList.append(game["gameId"])
  
        for i in range(len(gameList)):
             
            url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameList[i]}.json"
   
            response = requests.get(url)
            if response.status_code != 200:
                return {"status":"error", "msg":"today's games haven't started can't add to db"}
                
            data = response.json()

            
            time = data["game"]["gameTimeLocal"] 
            home = data["game"]["homeTeam"]["teamTricode"] 
            away = data["game"]["awayTeam"]["teamTricode"]
            
            statKeys = data["game"]["homeTeam"]["statistics"].keys()
            homeStatRow = {"game_id": gameList[i], "team": home}
    
            awayStatRow = {"game_id": gameList[i], "team": away}
                    

            for key in statKeys:
                if key.lower() == "steamfieldgoalattempts": #bug in nba games 100-200
                    homeStatRow["teamfieldgoalattempts"] = data["game"]["homeTeam"]["statistics"].get(key,0)
                    awayStatRow["teamfieldgoalattempts"] = data["game"]["awayTeam"]["statistics"].get(key,0)
                else:
                    homeStatRow[key.lower()] = data["game"]["homeTeam"]["statistics"].get(key,0)
                    awayStatRow[key.lower()] = data["game"]["awayTeam"]["statistics"].get(key,0)
            
            
            gameRow = {"game_id": gameList[i], "home_team": home, "away_team":away, "game_date":time}
     
            gameRows.append(gameRow)
            statRows.append(homeStatRow)
            statRows.append(awayStatRow)
        
       
      
        upsert(client, TABLES[1], gameRows) 
        upsert(client, TABLES[2], statRows)
        return {"status":"success", "message":"Games and game stats updated"}
                    
    except Exception as e:
        return {
            "status":"error",
            "message": e
        }
    
def query_games_table(client, limit=10):
    """Query the games table and return results as DataFrame"""
    try:
        response = client.table('games').select('*').limit(limit).execute()
        
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

def query_gamestats_table(client, limit=10):
    """Query the gamestats table and return results as DataFrame"""
    try:
        response = client.table('gamestats').select('*').limit(limit).execute()
        
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

def main():
    """
    TO DO:
        - Finish get getBoxScores 
    """

    load_dotenv()
    url = os.getenv("DB_URL")
    key = os.getenv("DB_KEY"); 
    client = create_client(url,key)
    args = sys.argv

    if len(args) > 2:
        print("Too many arguments")
    
    elif len(args) == 1:

        response = getTodayGames(client)
        print(response)
   
    elif args[1] == "all":
        getAllGames(client,SEASONS["REGULAR"],24)
        
    





if __name__ == "__main__":
    main()
