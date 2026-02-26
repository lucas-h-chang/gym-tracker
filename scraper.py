import requests
import sqlite3
from datetime import datetime

# ==========================================
# PHASE 1: GET THE DATA
# ==========================================
url = "https://api.density.io/v2/spaces/spc_863128347956216317/count"
headers = {
    "Authorization": "Bearer shr_o69HxjQ0BYrY2FPD9HxdirhJYcFDCeRolEd744Uj88e"
}

response = requests.get(url, headers=headers)
data = response.json()

current_count = data['count']
max_capacity = 150
percentage = int((current_count / max_capacity) * 100)

# Get the exact date and time right now
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{current_time}] Fetched data: {current_count} people ({percentage}%)")

# ==========================================
# PHASE 2: SAVE THE DATA
# ==========================================

# 1. Connect to a database file (Python will create 'gym_history.db' for you automatically)
conn = sqlite3.connect("gym_history.db")
cursor = conn.cursor()

# 2. Create our table (We use 'IF NOT EXISTS' so it only creates the headers the very first time you run the code)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS capacity_log (
        timestamp TEXT,
        people_count INTEGER,
        percent_full INTEGER
    )
''')

# 3. Insert our new row of data into those columns
cursor.execute('''
    INSERT INTO capacity_log (timestamp, people_count, percent_full)
    VALUES (?, ?, ?)
''', (current_time, current_count, percentage))

# 4. Save our changes and close the file
conn.commit()
conn.close()

print("Success! Data saved to the database.")