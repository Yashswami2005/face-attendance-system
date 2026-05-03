import sqlite3

conn = sqlite3.connect("attendance.db")
c = conn.cursor()

# ✅ Create students table
c.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no TEXT,
    name TEXT,
    branch TEXT,
    division TEXT
)
""")

# ✅ Create attendance table
c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no TEXT,
    name TEXT,
    branch TEXT,
    division TEXT,
    date TEXT,
    hour TEXT,
    time TEXT
)
""")

conn.commit()
conn.close()
print("✅ Database and tables created successfully!")
