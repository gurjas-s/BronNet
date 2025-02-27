import os
from supabase import create_client, Client
from dotenv import load_dotenv, dotenv_values
import requests


SEASONS = {"REGULAR":2, "PLAYOFFS":5}
MAX_GAMES = 1230
TABLES = {1:"games", 2:"gamestats"}

def makeDBConnection():
    load_dotenv()
    url = os.getenv("DB_URL")
    key = os.getenv("DB_KEY"); 
    return create_client(url,key)

"""
The NBA's Game ID, 0021400001, is a 10-digit code: XXXYYGGGGG, where XXX refers to a season prefix, YY is the season year (e.g. 14 for 2014-15), and GGGGG refers to the game number (1-1230 for a full 30-team regular season).
"""

def getAllGameIds(seasonType, year):

    seasonDigit = 24
    return gameList

        
def upsert(client, tableName, rows):
    try:
        response = (

            client.table(tableName).upsert(rows).execute()
        )
        return response
    except Exception as e:
        
        print(f"Error during upsert for {tableName}: {error}")
        raise
 
def getAllGames(client, seasonType, year):
    gameList = [] #Stores all game Id's for the given season type and year
    gameRows = []
    statRows = []
    start = 534
    batchSize = 100
    for i in range(1,MAX_GAMES):
        gameId = f"00{seasonType}{year}{i:05d}"
        gameList.append(gameId)
    
    for i in range(start,len(gameList)):
        
        if len(gameRows)>=batchSize:
            try:
                upsert(client,TABLES[1], gameRows) 
                upsert(client, TABLES[2], statRows)
                gameRows = []
                statRows = []
                print("Added 100 games")
            except Exception as error:
                print(f"Error: {error}")
                return 

        url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameList[i]}.json"
        response = requests.get(url)
        if response.status_code != 200:
            continue
        data = response.json()


        time = data["game"]["gameTimeLocal"] 
        home = data["game"]["homeTeam"]["teamTricode"] 
        away = data["game"]["awayTeam"]["teamTricode"]
        homeStats = data["game"]["homeTeam"]["statistics"]
        awayStats = data["game"]["awayTeam"]["statistics"]

        gameRow = {"game_id": gameList[i], "home_team": home, "away_team":away, "game_date":time}
        homeStatRow = {"game_id": gameList[i], "team": home, 
                "team_stats": homeStats}
        awayStatRow = {"game_id": gameList[i], "team": away, 
                "team_stats": awayStats}
        
        gameRows.append(gameRow)
        statRows.append(homeStatRow)
        statRows.append(awayStatRow)
        print(f"Added game {i} to arrays")
    
    try:
        upsert(client,TABLES[1], gameRows) 
        upsert(client, TABLES[2], statRows)
        gameRows = []
        statRows = []
        print("Added last amount of games")
    except Exception as error:
        print("Error: {error}")
        return 

   

def main():
    """
    TO DO:
        - Finish get getBoxScores 
    """
    client = makeDBConnection() 
    getAllGames(client, SEASONS["REGULAR"], 24) 
    
    

     
     


if __name__ == "__main__":
    main()
