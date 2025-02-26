import os
from supabase import create_client, Client
from dotenv import load_dotenv, dotenv_values
import requests

def makeDBConnection():
    load_dotenv()
    url = os.getenv("DB_URL")
    key = os.getenv("DB_KEY"); 
    return create_client(url,key)

def main():
    """
    TO DO:
        - Use requests to get data and update DB with necessary information
    """
    client = makeDBConnection() 
       


if __name__ == "__main__":
    main()
