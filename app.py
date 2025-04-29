from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import joblib
import pandas as pd

import database
import db_connection
import preprocessing
import training

app = FastAPI()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODELS_DIR = "models" 
os.makedirs(MODELS_DIR, exist_ok=True)  
# === Problem 2 ===

@app.post("/train/problem_2")
def train_problem_2():
    try:
        db = database.Database(db_connection.DbConnection().DATABASE_URL)
        df2, _, _ = db.collect_data()

        preprocessor = preprocessing.Preprocessor(df2, problem_type="problem_2")
        df_processed = preprocessor.preprocess_features()

        trainer = training.Training(df_processed)
        clustered_df, model = trainer.train_model([
            'avg_unit_price', 'total_sales_count', 'avg_quantity_per_order', 'unique_customer_count'
        ])


        output_path = os.path.join(OUTPUT_DIR, "dbscan_clustered_products_problem_2.csv")
        clustered_df.to_csv(output_path, index=False)

        return {
            "message": "Problem 2 modeli eğitildi.",
            "file": output_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")
@app.get("/download/problem_2")
def download_problem_2():
    file_path = os.path.join(OUTPUT_DIR, "dbscan_clustered_products_problem_2.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model çıktısı bulunamadı.")
    return FileResponse(file_path, media_type='text/csv', filename='dbscan_clustered_products_problem_2.csv')


# === Problem 3 ===


@app.post("/train/problem_3")
def train_problem_3():
    try:
        db = database.Database(db_connection.DbConnection().DATABASE_URL)
        _,df3, _  = db.collect_data()

        preprocessor = preprocessing.Preprocessor(df3, problem_type='problem_3')
        df_processed = preprocessor.preprocess_features()

        feature_columns = ['supplied_product_count', 'total_quantity_sold', 'average_unit_price', 'order_count']
        trainer = training.Training(df_processed)
        clustered_df,model = trainer.train_model(feature_columns)

        output_path = os.path.join(OUTPUT_DIR, "dbscan_clustered_suppliers_problem_3.csv")
        clustered_df.to_csv(output_path, index=False)

        return {
            "message": "Problem 3 modeli eğitildi.",
            "file": output_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")

@app.get("/download/problem_3")
def download_problem_3():
    file_path = os.path.join(OUTPUT_DIR, "dbscan_clustered_suppliers_problem_3.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model çıktısı bulunamadı.")
    return FileResponse(file_path, media_type='text/csv', filename='dbscan_clustered_suppliers_problem_3.csv')


# === Problem 4 ===

@app.post("/train/problem_4")
def train_problem_4():
    try:
        db = database.Database(db_connection.DbConnection().DATABASE_URL)
        _,_,df4 = db.collect_data()

        preprocessor = preprocessing.Preprocessor(df4, problem_type='problem_4')
        df_processed = preprocessor.preprocess_features()

        feature_columns = ['total_orders', 'avg_order_amount', 'avg_products_per_order']
        trainer = training.Training(df_processed)
        clustered_df,model = trainer.train_model(feature_columns)

        output_path = os.path.join(OUTPUT_DIR, "dbscan_clustered_customers_problem_4.csv")
        clustered_df.to_csv(output_path, index=False)

        return {
            "message": "Problem 4 modeli eğitildi.",
            "file": output_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")

@app.get("/download/problem_4")
def download_problem_4():
    file_path = os.path.join(OUTPUT_DIR, "dbscan_clustered_customers_problem_4.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model çıktısı bulunamadı.")
    return FileResponse(file_path, media_type='text/csv', filename='dbscan_clustered_customers_problem_4.csv')
