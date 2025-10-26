from fastapi import FastAPI, Query, Path
from typing import Optional
from datetime import date
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
load_dotenv()

PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

print("PostgreSQL Connection Settings:")
print(f"Host: {PGHOST}")
print(f"Port: {PGPORT}")
print(f"Database: {PGDATABASE}")
print(f"User: {PGUSER}")
print(f"Password: {'[SET]' if PGPASSWORD else '[NOT SET]'}")

connection_string = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
print("Connection psql string:", connection_string)

engine = create_engine(connection_string, pool_pre_ping=True)

app = FastAPI()

sector_map = {
    'AIRTEL': 'Telecommunication',
    'BHL': 'Hospitality',
    'FDHB': 'Financial',
    'FMBCH': 'Financial',
    'ICON': 'Construction',
    'ILLOVO': 'Agriculture',
    'MPICO': 'Construction',
    'NBM': 'Financial',
    'NBS': 'Financial',
    'NICO': 'Financial',
    'NITL': 'Financial',
    'OMU': 'Financial',
    'PCL': 'Investments',
    'STANDARD': 'Financial',
    'SUNBIRD': 'Hospitality',
    'TNM': 'Telecommunication'
}

#First end-point: with Query parameters

# Root endpoint
@app.get("/", summary="Root Endpoint")
def root():
    return {"message": "API is running"}

@app.get("/", summary="Root Endpoint")
def root():
    return {"message": "API is running"}

@app.get("/companies")
def get_counters_by_sector(sector: Optional[str] = Query(None, description="Filter counters by sector")):
    if sector:
        query = text("SELECT ticker, name, sector, date_listed FROM counters WHERE LOWER(sector) = LOWER(:sector);")
        params = {"sector": sector}
    else:
        query = text("SELECT ticker, name, sector, date_listed FROM counters;")
        params = {}
    
    with engine.connect() as conn:
        result = conn.execute(query, params).fetchall()
        print(result)
        counters = [
            {"ticker": row[0], "name": row[1], "sector": row[2], "date_listed": row[3]}
            for row in result
        ]
    
    if not counters:
        return {"message": f"No counters found for sector '{sector}'." if sector else "No counters found."}
    
    return {"sector": sector, "count": len(counters), "results": counters}

#SECOND ENDPOINT
@app.get("/companies/{ticker}")
def get_company_details(ticker: str):
    try:
        # Counter details - filter in SQL instead of loading all data
        query1 = text("SELECT counter_id, ticker, name, sector, date_listed FROM counters WHERE ticker = :ticker")
        
        with engine.connect() as conn:
            result1 = conn.execute(query1, {"ticker": ticker}).fetchone()
            
            if result1 is None:
                return {"error": f"Company with ticker '{ticker}' not found"}
            
            counter_id = result1[0]
            company_details = [{
                "ticker": result1[1],
                "name": result1[2],
                "sector": result1[3],
                "date_listed": result1[4]
            }]
            
            # Get counter records from daily prices
            query2 = text("SELECT COUNT(*) FROM prices_daily WHERE counter_id = :counter_id")
            result2 = conn.execute(query2, {"counter_id": counter_id}).fetchone()
            records = result2[0] if result2 else 0
        
        return {
            'Company details': company_details,
            'Total records': records
        }
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


#Third end-point

@app.get("/prices/daily")
def get_daily_prices(
    ticker: str = Query(..., description="Stock ticker symbol"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(100, description="Maximum records to return"),
):
    """
    Retrieve daily stock prices for a given company (by ticker),
    optionally filtered by date range and record limit.
    """
    # --- Fetch counter ID ---
    query_counter = text("SELECT counter_id FROM counters WHERE ticker = :ticker;")
    with engine.connect() as conn:
        result = conn.execute(query_counter, {"ticker": ticker}).fetchone()
        if not result:
            return {"error": f"Company with ticker '{ticker}' not found."}
        counter_id = result[0]

    # --- Retrieve daily price data for the counter ---
    query_prices = text("""
        SELECT open_mwk, high_mwk, low_mwk, close_mwk, volume, trade_date
        FROM prices_daily
        WHERE counter_id = :counter_id
        ORDER BY trade_date DESC;
    """)
    df = pd.read_sql(query_prices, con=engine, params={"counter_id": counter_id})
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'trade_date']

    # --- Ensure trade_date is a Timestamp ---
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')

    # --- Filter by date range safely ---
    if start_date:
        df = df[df['trade_date'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['trade_date'] <= pd.Timestamp(end_date)]

    # --- Apply record limit (max 1000) ---
    df = df.head(min(limit or 100, 1000))

    # --- Replace NaN with empty strings ---
    df = df.fillna('')

    return {
        "Company": ticker,
        "Total records": len(df),
        "data": df.to_dict(orient='records')
    }


#Fouth end-point
@app.get("/prices/range")
def get_prices_by_period(
    ticker: str = Query(..., description="Stock ticker symbol"),
    year: int = Query(..., description="Year"),
    month: Optional[int] = Query(None, description="Month of the year"),
):
    """
    Retrieve stock prices for a given company, filtered by year
    and optionally by month.
    """

    # --- Fetch counter ID ---
    query_counter = text("SELECT counter_id FROM counters WHERE ticker = :ticker;")
    with engine.connect() as conn:
        result = conn.execute(query_counter, {"ticker": ticker}).fetchone()
        if not result:
            return {"error": f"Company with ticker '{ticker}' not found."}
        counter_id = result[0]

    # --- Build SQL filter ---
    date_filter = "EXTRACT(YEAR FROM trade_date) = :year"
    params = {"counter_id": counter_id, "year": year}
    if month:
        date_filter += " AND EXTRACT(MONTH FROM trade_date) = :month"
        params["month"] = month

    # --- Retrieve data ---
    query_prices = text(f"""
        SELECT trade_date AS period,
               open_mwk AS open,
               high_mwk AS high,
               low_mwk AS low,
               close_mwk AS close,
               volume AS total_volume
        FROM prices_daily
        WHERE counter_id = :counter_id AND {date_filter}
        ORDER BY trade_date DESC;
    """)
    df = pd.read_sql(query_prices, con=engine, params=params)

    # --- Handle empty results safely ---
    if df.empty:
        return {
            "Company": ticker,
            "Year": year,
            "Month": month,
            "Total records": 0,
            "data": []
        }

    # --- Ensure consistent datetime column ---
    df['period'] = pd.to_datetime(df['period'], errors='coerce')
    df = df.fillna("")

    return {
        "Company": ticker,
        "Year": year,
        "Month": month,
        "Total records": len(df),
        "data": df.to_dict(orient="records")
    }


#Fith end-point

@app.get("/prices/latest")
def get_recent_prices(ticker: str = Query(..., description="Stock ticker symbol")):
    """
    Retrieve the latest stock price for a given company,
    along with the previous price and price change (absolute and percentage).
    """

    # --- Fetch counter ID for the given ticker ---
    query_counter = text("SELECT counter_id FROM counters WHERE ticker = :ticker;")
    with engine.connect() as conn:
        result = conn.execute(query_counter, {"ticker": ticker}).fetchone()
        if not result:
            return {"error": f"Company with ticker '{ticker}' not found."}
        counter_id = result[0]

    # --- Fetch all daily prices for this counter ---
    query_prices = text("""
        SELECT trade_date, open_mwk, high_mwk, low_mwk, close_mwk, volume
        FROM prices_daily
        WHERE counter_id = :counter_id
    """)
    df = pd.read_sql(query_prices, con=engine, params={"counter_id": counter_id})

    if df.empty:
        return {"error": f"No price data found for ticker '{ticker}'."}

    # --- Clean and rename columns ---
    df.columns = ['trade_date', 'open', 'high', 'low', 'close', 'total_volume']
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # --- Sort descending by trade date to get latest price ---
    df_sorted = df.sort_values(by='trade_date', ascending=False).reset_index(drop=True)
    latest = df_sorted.iloc[0]
    latest_date = latest['trade_date']
    latest_price = latest['close']

    # --- Calculate previous price and change ---
    if len(df_sorted) > 1:
        prev_price = df_sorted.iloc[1]['close']
        change = latest_price - prev_price
        change_percentage = (change / prev_price * 100) if prev_price != 0 else 0
    else:
        prev_price = None
        change = None
        change_percentage = None

    return {
        "ticker": ticker,
        "latest_date": latest_date,
        "latest_price": latest_price,
        "previous_price": prev_price,
        "change": change,
        "change_percentage": f"{round(change_percentage, 3)}%" if change_percentage is not None else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.utils.mse_api:app", host="localhost", port=8000, reload=True)