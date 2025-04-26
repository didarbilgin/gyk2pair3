from sqlalchemy import create_engine
import pandas as pd


class Database:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        
    def collect_data(self):
        query = """select * from products p
        inner join order_details od
        on p.product_id = od.product_id
        """
        df = pd.read_sql(query, self.engine)
        return df

