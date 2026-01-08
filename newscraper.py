from dotenv import load_dotenv
import os
import requests
import pandas as pd
import time

load_dotenv()

API_KEY = os.getenv('NEWS_API_KEY')
BASE_URL = 'https://newsapi.org/v2/'

if not API_KEY:
    raise ValueError("NEWS_API_KEY not found")

def scrape_real_news(query, pages=10):
    articles = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}everything"
        params = {
            'q': query,
            'sources': 'bbc-news,reuters,associated-press',
            'apiKey': API_KEY,
            'page': page,
            'pageSize': 100,
            'language': 'en'
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error on page {page}: {response.status_code}")
            print(f"Response: {response.text}")
            break
        data = response.json()

        if 'articles' in data:
            for article in data['articles']:
                if article['title'] and article['content']:
                    articles.append({
                        'title': article['title'],
                        'text': article['content'] or article['description'],
                        'label': 'REAL'
                    })
        
        time.sleep(1)       # rate limiting checkpoint
        print(f"Scraped {page}. At {len(articles)} news articles.")

    return pd.DataFrame(articles)

real_news = scrape_real_news('politics', pages=20)
real_news.to_csv('scraped_real_news.csv', index=False)