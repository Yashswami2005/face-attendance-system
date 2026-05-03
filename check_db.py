import sqlite3

# Connect to your database
conn = sqlite3.connect("attendance.db")
c = conn.cursor()

# Show structure of 'attendance' table
c.execute("PRAGMA table_info(attendance)")
columns = c.fetchall()

print("📋 Columns in attendance table:")
for col in columns:
    print(f"{col[1]} ({col[2]})")

conn.close()
