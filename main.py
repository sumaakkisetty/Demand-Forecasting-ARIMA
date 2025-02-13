import pymongo
import pandas as pd
import numpy as np
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

warnings.filterwarnings("ignore")

# ----------------- DemandForecaster Class ----------------- #

class DemandForecaster:
    def __init__(self, connection_string: str, database_name: str):
        """Initialize the DemandForecaster with MongoDB connection details."""
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[database_name]
        self.sales_collection = self.db['sales']

    def fit_single_arima(self, ts: pd.Series, order=(1, 1, 1)):
        """Fit a single ARIMA model."""
        try:
            model = ARIMA(ts, order=order)
            model_fit = model.fit()
            return model_fit
        except Exception as e:
            print(f"Error fitting ARIMA{order}: {str(e)}")
            return None

    def forecast_weekly_demand(self, warehouse_id: str, product_id: str):
        """Forecast next week's demand for a given product at a specified warehouse."""
        try:
            pipeline = [
                {"$unwind": "$sales"},
                {"$match": {"sales.warehouse_id": warehouse_id, "sales.product_id": product_id}},
                {"$group": {"_id": "$week", "weekly_sales": {"$sum": "$sales.quantity_sold"}}},
                {"$sort": {"_id": 1}}
            ]

            data = list(self.sales_collection.aggregate(pipeline))
            if not data:
                raise ValueError(f"No sales data found for warehouse: {warehouse_id}, product: {product_id}")

            df = pd.DataFrame(data)
            df['_id'] = pd.to_datetime(df['_id'])
            df = df.sort_values('_id').set_index('_id')
            ts = df['weekly_sales'].fillna(method='ffill').fillna(method='bfill')

            orders = [(1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)]
            best_model, best_aic, best_order = None, np.inf, None

            for order in orders:
                model_fit = self.fit_single_arima(ts, order)
                if model_fit and model_fit.aic < best_aic:
                    best_model, best_aic, best_order = model_fit, model_fit.aic, order

            if best_model is None:
                raise ValueError("Could not fit any ARIMA model")

            forecast = best_model.forecast(steps=1)[0]

            return {
                'warehouse_id': warehouse_id,
                'product_id': product_id,
                'forecast_date': (ts.index[-1] + pd.Timedelta(weeks=1)).strftime('%Y-%m-%d'),
                'forecast_value': int(forecast),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def forecast_all(self):
        """Forecast next week's demand for all warehouse-product combinations."""
        try:
            combinations_pipeline = [
                {"$unwind": "$sales"},
                {"$group": {"_id": {"warehouse_id": "$sales.warehouse_id", "product_id": "$sales.product_id"}}}
            ]
            combinations = list(self.sales_collection.aggregate(combinations_pipeline))
            results = []

            for combo in combinations:
                warehouse_id = combo['_id']['warehouse_id']
                product_id = combo['_id']['product_id']
                try:
                    forecast_result = self.forecast_weekly_demand(warehouse_id, product_id)
                    results.append(forecast_result)
                except Exception:
                    continue  # Skip if forecasting fails

            return results

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# ----------------- FastAPI Application ----------------- #

app = FastAPI(title="Demand Forecasting API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Credentials from Render Environment Variables
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "your_default_connection_string_here")
DATABASE_NAME = os.getenv("DATABASE_NAME", "InvenX")

# Initialize Forecaster
forecaster = DemandForecaster(MONGO_CONNECTION_STRING, DATABASE_NAME)

@app.get("/forecast-all")
def get_forecast_all():
    """API to forecast demand for all products in all warehouses."""
    try:
        results = forecaster.forecast_all()
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Run the API Server ----------------- #

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
