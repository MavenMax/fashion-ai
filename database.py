import sqlite3

conn = sqlite3.connect("fashion_ai.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    email TEXT PRIMARY KEY,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_data(
    email TEXT,
    name TEXT,
    age INTEGER,
    occasion TEXT,
    weather TEXT,
    fav_brand TEXT
)
""")

conn.commit()