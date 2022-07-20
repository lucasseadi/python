import hashlib
import sqlite3


class Back:
    DATABASE = "login.sqlite"

    def __init__(self):
        try:
            r = open("startDB.txt", encoding="UTF-8")
            data = r.readlines()

            conn = sqlite3.connect(self.DATABASE)
            c = conn.cursor()

            for i in data:
                # print(i)
                c.execute(i)
                conn.commit()

            db_count = self.check_empty_database()
            if db_count[0][0] == 0:
                hashed_username = hashlib.md5("user1".encode("UTF-8")).hexdigest()
                hashed_password = hashlib.md5("user1".encode("UTF-8")).hexdigest()
                self.execute_db_query("INSERT INTO LOGINCREDENTIALS(USERNAME, PASSWORD) VALUES (?, ?)",
                            (hashed_username, hashed_password))
                print("user1 added")

            conn.close()
        except Exception as e:
            raise

    def execute_db_query(self, query, parameters=()):
        with sqlite3.connect(self.DATABASE) as conn:
            print(conn)
            print("You have successfully connected to the database.")
            cursor = conn.cursor()
            query_result = cursor.execute(query, parameters)
            conn.commit()
        return query_result

    # Checks for empty database
    def check_empty_database(self):
        return self.select("SELECT COUNT(*) FROM LOGINCREDENTIALS")

    def select(self, query, parameters=()):
        conn = sqlite3.connect(self.DATABASE)
        c = conn.cursor()
        c.execute(query, parameters)
        result = c.fetchall()
        conn.commit()
        conn.close()
        return result

