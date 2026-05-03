import sqlite3

# Database file name
DB = "attendance.db"

def create_table():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # ✅ Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database and table ready!")

if __name__ == "__main__":
    create_table()
