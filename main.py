import os
from supabase import create_client, Client
from dotenv import load_dotenv, dotenv_values
import requests

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
    gameList = []
    for i in range(1,1230):
        gameId = f"00{seasonType}{seasonDigit}{i:05d}"
        gameList.append(gameId)
    return gameList

def getAllBoxScores():
    
    gameIds = getAllGameIds(2, 24)
    gameRows = []
    for i in range(50):
        url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameIds[i]}.json"  
        data = requests.get(url).json()
        if data is None:
            break
        time = data["game"]["gameTimeLocal"] 
        home = data["game"]["homeTeam"]["teamTricode"] 
        away = data["game"]["awayTeam"]["teamTricode"]
        gameRow = {"gameId": gameIds[i], "home_team": home, "away_team":away, "game_date":time}
        gameRows.append(gameRow) 
        
    
    print(gameRows)
        
     
    

def main():
    """
    TO DO:
        - Finish get getBoxScores 
    """
    getAllBoxScores()
    #client = makeDBConnection() 
       


if __name__ == "__main__":
    main()
