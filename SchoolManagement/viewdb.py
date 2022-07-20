import sqlite3

DATABASE = "login.sqlite"
conn = sqlite3.connect(DATABASE)
c = conn.cursor()
c.execute("SELECT * FROM COURSES")
result = c.fetchall()
for row in result:
    print(row)
conn.close()