import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

class Database:
    def __init__(self, database_url):
        try:
            self.engine = create_engine(database_url)
            self.connection = self.engine.connect()
            print("[INFO] Database bağlantısı başarılı.")
        except SQLAlchemyError as e:
            print(f"[ERROR] Database bağlantı hatası: {str(e)}")
            self.connection = None

    def collect_data(self):
        if self.connection is None:
            raise Exception("Database bağlantısı başarısız. Veri çekilemiyor.")

        query2 = """
        SELECT
        p.product_id,
        AVG(od.unit_price) AS avg_unit_price,
        COUNT(od.order_id) AS total_sales_count,
        AVG(od.quantity) AS avg_quantity_per_order,
        COUNT(DISTINCT o.customer_id) AS unique_customer_count
        FROM
        products p
        INNER JOIN order_details od ON p.product_id = od.product_id
        INNER JOIN orders o ON od.order_id = o.order_id
        GROUP BY
        p.product_id
        """

        query3 = """
        SELECT
        suppliers.supplier_id,
        suppliers.company_name,
        products.product_id,
        order_details.quantity,
        order_details.unit_price,
        order_details.order_id
        FROM suppliers
        JOIN products ON suppliers.supplier_id = products.supplier_id
        JOIN order_details ON products.product_id = order_details.product_id
        """

        query4 = """
        SELECT
            c.country,
            COUNT(DISTINCT o.order_id) AS total_orders,
            AVG(od.unit_price * od.quantity) AS avg_order_value,
            AVG(order_item_count.item_count) AS avg_items_per_order
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN (
            SELECT order_id, COUNT(*) AS item_count
            FROM order_details
            GROUP BY order_id
        ) AS order_item_count ON o.order_id = order_item_count.order_id
        JOIN order_details od ON o.order_id = od.order_id
        GROUP BY c.country
        """

        df2 = pd.read_sql_query(query2, self.connection)
        df3 = pd.read_sql_query(query3, self.connection)
        df4 = pd.read_sql_query(query4, self.connection)
        return df2, df3, df4

