import os
from supabase import create_client, Client
from dotenv import load_dotenv, dotenv_values


def main():
    load_dotenv()
    url = os.getenv("DB_URL")
    if not url:
        print(url)
        print("Error")
    key = os.getenv("DB_KEY");
    #url: str = os.environ.get("DB_URL")
    #key: str = os.environ.get("DB_KEY")
    #supabase: Client = create_client(url, key)
    client = create_client(url,key)
    print("hello world")


if __name__ == "__main__":
    main()
