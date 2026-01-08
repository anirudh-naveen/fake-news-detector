import pandas as pd
existing = pd.read_csv('news.csv')
scraped = pd.read_csv('scraped_real_news.csv')
combined = pd.concat([existing, scraped], ignore_index=True)
combined = combined.sample(frac=1).reset_index(drop=True)
combined.to_csv('combined_news.csv', index=False)