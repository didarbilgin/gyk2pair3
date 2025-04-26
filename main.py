import database
import db_connection
import pandas as pd

class Main:
    def __init__(self):
        self.database_url = db_connection.DbConnection().DATABASE_URL  # DbConnection sınıfından DATABASE_URL'i alıyoruz
        self.db = database.Database(database_url)  # Database sınıfından bir nesne oluşturuyoruz

    def run(self):
        df = self.db.collect_data()  # Database içindeki collect_data fonksiyonunu çağırıyoruz
        print(df)

if __name__ == "__main__":
    app = Main()
    app.run()