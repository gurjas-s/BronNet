from flask import Flask, jsonify
from db import makeDBConnection, getTodayGames
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

def updateGames(): #Automatically run once a day
    try:
        client = makeDBConnection()
        response = getTodayGames(client)
        print("Updated today's games")
    except Exception as e:
        print(f"Error updating games: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(updateGames, 'interval', days=1)
scheduler.start()

@app.route('/updateGames', methods=['POST'])


def update(): #On request

    client = makeDBConnection()
    response = getTodayGames(client)
    return jsonify(response)

        
if __name__ == '__main__':
    app.run()
