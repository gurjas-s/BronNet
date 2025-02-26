import os
from supabase import create_client, Client
from dotenv import load_dotenv, dotenv_values
import requests


seasons = {"REGULAR":2, "PLAYOFFS":5}

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

def getAllGames(seasonType, year):
    gameList = [] #Stores all game Id's for the given season type and year
    gameRows = []
    statRows = []
    for i in range(1,1230):
        gameId = f"00{seasonType}{year}{i:05d}"
        gameList.append(gameId)
    
    for i in range(50):
        url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameList[i]}.json"  
        data = requests.get(url).json()
        if data is None:
            break
        time = data["game"]["gameTimeLocal"] 
        home = data["game"]["homeTeam"]["teamTricode"] 
        away = data["game"]["awayTeam"]["teamTricode"]
        gameRow = {"gameId": gameList[i], "home_team": home, "away_team":away, "game_date":time}
        gameRows.append(gameRow) 
         
    return gameRows 
        
     
    

def main():
    """
    TO DO:
        - Finish get getBoxScores 
    """
    games, boxscores = getAllGames(seasons["REGULAR"], 24) 
    print(games)
    #client = makeDBConnection() 
       


if __name__ == "__main__":
    main()
