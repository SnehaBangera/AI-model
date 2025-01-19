import requests
import os

def fetch_stock_news_alpha_vantage(stock_symbol):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}&apikey={api_key}'
    
    try:
        response = requests.get(url)

        if response.status_code != 200:
            return []

        data = response.json()
        
        if 'feed' not in data:
            return []
        
        news_list = []
        for article in data['feed'][:5]:  # Limit to top 5 articles
            news_list.append({
                "title": article['title'],
                "description": article['summary'],
                "url": article['url']
            })
        
        return news_list

    except requests.exceptions.RequestException:
        return []
