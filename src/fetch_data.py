import yfinance as yf
from datetime import datetime

def fetch_data(stock_symbol, start_date="2008-01-01", end_date=None, interval='1d'):
    if not stock_symbol or not isinstance(stock_symbol, str):
        raise ValueError("Invalid stock symbol provided.")
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            raise ValueError("No data retrieved for the given stock symbol and date range.")
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
