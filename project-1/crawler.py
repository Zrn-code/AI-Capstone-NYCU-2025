import yfinance as yf
import pandas as pd
import datetime
import time

import feedparser
from bs4 import BeautifulSoup
import urllib
from dateparser import parse as parse_date
import requests


def _parse_sub_articles(text):
    """Parse subarticles from article summaries"""
    try:
        bs4_html = BeautifulSoup(text, "html.parser")
        lis = bs4_html.find_all('li')
        sub_articles = []
        for li in lis:
            try:
                sub_articles.append({
                    "url": li.a['href'],
                    "title": li.a.text,
                    "publisher": li.font.text
                })
            except:
                pass
        return sub_articles
    except:
        return text


def search_google_news(query: str, lang='en', country='US', helper=True, date=None, proxies=None, scraping_bee=None):
    """
    Search Google News and return a list of articles.
    
    :param str query: Search query
    :param str lang: Language code (default: 'en')
    :param str country: Country code (default: 'US')
    :param bool helper: When True helps with URL quoting
    :param str when: Sets a time range for the articles that can be found
    :param dict proxies: Optional proxies configuration
    :param str scraping_bee: Optional ScrapingBee API key
    :return: Dictionary with feed info and entries
    """
    # Convert parameters
    lang = lang.lower()
    country = country.upper()
    BASE_URL = 'https://news.google.com/rss'
    
    # Add time range if specified
    if date:
        
        after_date = parse_date(date) - datetime.timedelta(days=1)
        query += f' after:{after_date.strftime("%Y-%m-%d")}'
        query += f' before:{date}'

    # URL quote the query if helper is enabled
    if helper == True:
        query = urllib.parse.quote_plus(query)
        
    # Build the CEID parameter
    ceid = f'?ceid={country}:{lang}&hl={lang}&gl={country}'
    search_ceid = ceid.replace('?', '&')
    
    # Build the feed URL
    feed_url = f'{BASE_URL}/search?q={query}{search_ceid}'
    
    # Check if both ScrapingBee and proxies are specified
    if scraping_bee and proxies:
        raise Exception("Pick either ScrapingBee or proxies. Not both!")
        
    # Get the feed content
    if proxies:
        r = requests.get(feed_url, proxies=proxies)
    elif scraping_bee:
        # Using regular request as fallback since scraping_bee implementation is missing
        r = requests.get(feed_url)
    else:
        r = requests.get(feed_url)
        
    if 'https://news.google.com/rss/unsupported' in r.url:
        raise Exception('This feed is not available')
        
    # Parse the feed
    d = feedparser.parse(r.text)
    
    if not scraping_bee and not proxies and len(d['entries']) == 0:
        d = feedparser.parse(feed_url)
        
    result = dict((k, d[k]) for k in ('feed', 'entries'))
    
    # Add sub-articles to each entry
    for i, val in enumerate(result['entries']):
        if 'summary' in result['entries'][i].keys():
            result['entries'][i]['sub_articles'] = _parse_sub_articles(result['entries'][i]['summary'])
        else:
            result['entries'][i]['sub_articles'] = None
            
    return result

# Fetch stock market data
def fetch_stock_data(ticker="^GSPC", num_days=100, stock_output="stock_data.csv"):
    """ Fetch stock market data for the last num_days days and save as CSV (format: date,label,change). """
    data = yf.download(ticker, period=f"{num_days}d")
    data['Change'] = data['Close'] - data['Open']  # Calculate daily change
    def get_label(change):
        if change >= 0:
            return 'up'
        else:
            return 'down'

    data['label'] = data['Change'].apply(get_label)

    # Reformat DataFrame - keep Change column
    data = data[['Change', 'label']]
    data.index = data.index.strftime('%Y-%m-%d')  # Ensure date format is correct
    data.index.name = 'date'  # Set index name
    data.reset_index(inplace=True)  # Make 'date' a regular column
    # Save as CSV
    data.to_csv(stock_output, index=False)
    print(f"Stock market data saved to {stock_output}")
    return stock_output


# Fetch news data
def fetch_news_for_days(num_days=100, news_output="news_data.csv"):
    """ Fetch Google News articles and save as CSV """
    date_today = datetime.datetime.now().date()
    dates = [date_today - datetime.timedelta(days=i) for i in range(num_days)]
    news_data = []
    
    # Track seen headlines to avoid duplicates
    seen_headlines = set()
    # Counter to reset seen_headlines every 10 days
    days_processed = 0

    # Define category column names
    category_columns = {
        "Technology": "tech",
        "Business": "business", 
        "Market": "market",
        "Economic": "economy",
        "World": "events"
    }

    for date in dates:
        # Reset seen_headlines every 10 days
        if days_processed % 10 == 0 and days_processed > 0:
            print(f"Clearing seen headlines cache after {days_processed} days")
            seen_headlines.clear()
        
        days_processed += 1
        date_str = date.strftime('%Y-%m-%d')
        print(f"Getting news for {date_str}...")
        
        # Initialize data row
        row = {'date': date_str}
        
        # Search each category sequentially
        for category in category_columns.keys():
            try:
                print(f"  Searching {category} News...")
                
                # Use function to replace category
                search = search_google_news(category, date=date_str)
                
                # Get four news items for this category
                column_name = category_columns[category]
                article_count = 0
                
                for entry in search['entries']:
                    # Check if we already have this headline
                    if entry['title'] not in seen_headlines and article_count < 4:
                        article_num = article_count + 1
                        row[f"{column_name}_{article_num}"] = entry['title']
                        seen_headlines.add(entry['title'])
                        article_count += 1
                    # If we've reached 4 unique headlines, break
                    if article_count >= 4:
                        break
                
                # Fill remaining slots if needed
                for i in range(article_count + 1, 5):
                    row[f"{column_name}_{i}"] = f"No additional unique {category} news found"
                
            except Exception as e:
                print(f"  Error when searching category '{category}': {e}")
                # Fill all slots with error message if exception occurs
                for i in range(1, 5):
                    row[f"{category_columns[category]}_{i}"] = f"Unable to get {category} news"
                time.sleep(0.1)  # Wait a bit longer after error
        
        news_data.append(row)
        
        # Avoid sending requests too quickly
        time.sleep(0.05)

    # Save as CSV
    news_df = pd.DataFrame(news_data)
    news_df.to_csv(news_output, index=False)
    print(f"News data saved to {news_output}")
    return news_output

# Combine stock labels and news
def combine_stock_news(stock_file, news_file, output_file="news_with_stock_labels.csv"):
    """ Combine stock market data and news data, output as CSV """
    # Read stock market data
    stock_df = pd.read_csv(stock_file)
    stock_df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)  # Fix date column name
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.strftime('%Y-%m-%d')  # Format date

    # Read news data
    news_df = pd.read_csv(news_file)
    news_df['date'] = pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d')  # Format date

    # Combine both
    merged_df = pd.merge(stock_df,news_df, on="date", how="inner")

    # Save as CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

def divide_resources_news(news_file, output_file="news_with_stock_labels.csv"):
    """ Divide news headlines and their sources into separate columns """
    print(f"Dividing news headlines and sources in {news_file}...")
    
    # Read the combined data file
    df = pd.read_csv(news_file)
    
    # Define the categories to process
    categories = ["tech", "business", "market", "economy", "events"]
    
    # For each category and its articles (1-4)
    for category in categories:
        for i in range(1, 5):
            column_name = f"{category}_{i}"
            resource_column = f"{column_name}_resource"
            
            # Skip if the column doesn't exist
            if column_name not in df.columns:
                continue
                
            # Create new columns for the resources
            df[resource_column] = ""
            
            # Process each row
            for index, row in df.iterrows():
                news_text = str(row[column_name])
                
                # Find the last occurrence of " - " which typically separates headline from source
                separator_pos = news_text.rfind(" - ")
                
                if separator_pos != -1:
                    # Split the headline and source
                    headline = news_text[:separator_pos].strip()
                    source = news_text[separator_pos + 3:].strip()  # +3 to skip the " - "
                    
                    # Update the dataframe
                    df.at[index, column_name] = headline
                    df.at[index, resource_column] = source
    
    # Save the updated data
    df.to_csv(output_file, index=False)
    print(f"Headlines and sources divided and saved to {output_file}")

# Main function
def main():
    num_days = 1000  # Adjust number of days as needed
 
    # 1. Fetch stock market data
    stock_file = fetch_stock_data(num_days=num_days)

    # 2. Fetch news data
    news_file = fetch_news_for_days(num_days=num_days)

    # 3. Combine news and stock labels
    stock_file = "stock_data.csv"
    news_file = "news_data.csv"
    output_file = "news_with_stock_labels.csv"
    combine_stock_news(stock_file, news_file, output_file)
    devided_file = "news_with_stock_labels_devided.csv"
    # 4. divide resources and news into two columns
    divide_resources_news(output_file, devided_file)
    
    
if __name__ == "__main__":
    main()
