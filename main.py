import joblib
import database
import db_connection
import preprocessing
import training
import visualization


class Main:
    def __init__(self):
        self.database_url = db_connection.DbConnection().DATABASE_URL
        self.db = database.Database(self.database_url)

    def process_problem(self, problem_type, feature_columns, model_filename, output_filename, x_col, y_col):
        # Veri çekimi
        df2, df3, df4 = self.db.collect_data()
        df_map = {
            'problem_2': df2,
            'problem_3': df3,
            'problem_4': df4
        }

        if problem_type not in df_map:
            print(f"[ERROR] Geçersiz problem tipi: {problem_type}")
            return

        df = df_map[problem_type]
        print(f"[INFO] Veritabanından {problem_type} verisi çekildi.")

        # Ön işleme
        preprocessor = preprocessing.Preprocessor(df, problem_type=problem_type)
        df_processed = preprocessor.preprocess_features()
        print(f"[INFO] {problem_type} için özellik mühendisliği tamamlandı.")

        # Model eğitimi ve kümeleme
        trainer = training.Training(df_processed)
        clustered_df, dbscan_model = trainer.train_model(feature_columns)

        # Modeli ve sonuçları kaydet
        joblib.dump(dbscan_model, model_filename)
        print(f"[INFO] Model kaydedildi: {model_filename}")

        clustered_df.to_csv(output_filename, index=False)
        print(f"[INFO] Sonuçlar kaydedildi: {output_filename}")

        # DBSCAN ile kümeleme tamamlandıktan sonra
        visualizer = visualization.Visualization(clustered_df)

        # Özelleştirilmiş parametrelerle plot_clusters fonksiyonunu çağırıyoruz.
        visualizer.plot_clusters(x_col=x_col, y_col=y_col, cluster_col='cluster', title=f'{problem_type} Kümeleme')

    def run(self):
        # List of problems to process
        problems = [
            {
                'problem_type': 'problem_2',
                'feature_columns': ['avg_unit_price', 'total_sales_count', 'avg_quantity_per_order', 'unique_customer_count'],
                'model_filename': 'models/dbscan_model2.pkl',
                'output_filename': 'dbscan_clustered_products.csv',
                'x_col': 'total_sales_count',
                'y_col': 'avg_unit_price'
            },
            {
                'problem_type': 'problem_3',
                'feature_columns': ['supplied_product_count', 'total_quantity_sold', 'average_unit_price', 'order_count'],
                'model_filename': 'models/dbscan_model3.pkl',
                'output_filename': 'supplier_clusters.csv',
                'x_col': 'total_quantity_sold',
                'y_col': 'average_unit_price'
            },
            {
                'problem_type': 'problem_4',
                'feature_columns': ['total_orders', 'avg_order_amount', 'avg_products_per_order'],
                'model_filename': 'models/dbscan_model4.pkl',
                'output_filename': 'country_clusters.csv',
                'x_col': 'total_orders',
                'y_col': 'avg_order_amount'
            }
        ]

        # Process each problem in the list
        for problem in problems:
            self.process_problem(
                problem_type=problem['problem_type'],
                feature_columns=problem['feature_columns'],
                model_filename=problem['model_filename'],
                output_filename=problem['output_filename'],
                x_col=problem['x_col'],
                y_col=problem['y_col']
            )


if __name__ == "__main__":
    app = Main()
    app.run()  # Bu şekilde, tüm problemler sırasıyla işlenir
