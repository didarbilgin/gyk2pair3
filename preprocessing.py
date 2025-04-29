import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, df, problem_type):
        self.df = df
        self.problem_type = problem_type.lower()

    def preprocess_features(self):
        print("[INFO] Preprocessing başladı...")

        # Eksik veri kontrolü ve doldurma
        if self.df.isnull().sum().sum() > 0:
            print("[WARNING] Eksik veri bulundu! Eksik değerler ortalama ile dolduruluyor.")
            self.df = self.df.fillna(self.df.mean(numeric_only=True))

        # Problem 2 (ürün satış performansı analizi - örnek problem olabilir)
        if self.problem_type == 'problem_2':
            feature_columns = ['avg_unit_price', 'total_sales_count', 'avg_quantity_per_order', 'unique_customer_count']
            id_column = 'product_id'

        # Problem 3: Tedarikçi Segmentasyonu
        elif self.problem_type == 'problem_3':
            self.df['supplied_product_count'] = self.df.groupby('supplier_id')['product_id'].transform('nunique')
            self.df['total_quantity_sold'] = self.df.groupby('supplier_id')['quantity'].transform('sum')
            self.df['average_unit_price'] = self.df.groupby('supplier_id')['unit_price'].transform('mean')
            self.df['order_count'] = self.df.groupby('supplier_id')['order_id'].transform('nunique')
            feature_columns = ['supplied_product_count', 'total_quantity_sold', 'average_unit_price', 'order_count']
            id_column = 'supplier_id'

        # Problem 4: Ülkelere Göre Satış Deseni Analizi
        elif self.problem_type == 'problem_4':
            # Sorgudan gelen sütun isimlerini Python tarafında uyarlıyoruz
            self.df.rename(columns={
                'avg_order_value': 'avg_order_amount',
                'avg_items_per_order': 'avg_products_per_order'
            }, inplace=True)

            feature_columns = ['total_orders', 'avg_order_amount', 'avg_products_per_order']
            id_column = 'country'


        else:
            raise ValueError(f"[ERROR] Geçersiz problem türü: {self.problem_type}")

        # Seçilen özellikleri al ve ölçekle
        feature_df = self.df[feature_columns]
        scaler = StandardScaler()
        feature_df_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_columns)

        # ID sütununu da ekle
        result_df = pd.concat([self.df[[id_column]].reset_index(drop=True), feature_df_scaled], axis=1)

        print(f"[INFO] Preprocessing tamamlandı. Problem: {self.problem_type}")
        return result_df
