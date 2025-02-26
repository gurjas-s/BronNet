import os
from supabase import create_client, Client
from dotenv import load_dotenv, dotenv_values
import requests

def makeDBConnection():
    load_dotenv()
    url = os.getenv("DB_URL")
    key = os.getenv("DB_KEY"); 
    return create_client(url,key)

def getBoxScores(): 
    url = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022000181.json" #Hard code for now 
    data = requests.get(url)
    print(data)

def main():
    """
    TO DO:
        - Finish get getBoxScores 
    """
    getBoxScores()
    #client = makeDBConnection() 
       


if __name__ == "__main__":
    main()
