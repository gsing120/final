import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QualitativeAnalysis")

class QualitativeAnalyzer:
    """
    Performs qualitative analysis of financial markets by analyzing fundamental data,
    news sentiment, market narratives, and macroeconomic factors.
    """
    
    def __init__(self):
        """Initialize the qualitative analyzer."""
        # Download NLTK resources if not already present
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def get_company_profile(self, ticker):
        """Get detailed company profile and business description."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            profile = {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'business_summary': info.get('longBusinessSummary', 'No business summary available.'),
                'website': info.get('website', ''),
                'full_time_employees': info.get('fullTimeEmployees', 0),
                'country': info.get('country', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
            
            return profile
        
        except Exception as e:
            logger.error(f"Error getting company profile for {ticker}: {e}")
            return {
                'name': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'business_summary': 'Error retrieving business summary.',
                'website': '',
                'full_time_employees': 0,
                'country': 'Unknown',
                'exchange': 'Unknown',
                'market_cap': 0,
                'currency': 'USD'
            }
    
    def get_financial_health_metrics(self, ticker):
        """Get detailed financial health metrics."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            # Extract key metrics
            metrics = {}
            
            # Liquidity ratios
            if not balance_sheet.empty:
                try:
                    current_assets = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else 0
                    current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else 0
                    
                    if current_liabilities != 0:
                        metrics['current_ratio'] = current_assets / current_liabilities
                    else:
                        metrics['current_ratio'] = 0
                        
                    # Quick ratio (Acid-test ratio)
                    inventory = balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else 0
                    if current_liabilities != 0:
                        metrics['quick_ratio'] = (current_assets - inventory) / current_liabilities
                    else:
                        metrics['quick_ratio'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating liquidity ratios: {e}")
                    metrics['current_ratio'] = 0
                    metrics['quick_ratio'] = 0
            
            # Solvency ratios
            if not balance_sheet.empty:
                try:
                    total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
                    total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else 0
                    total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
                    
                    if total_assets != 0:
                        metrics['debt_to_assets'] = total_liabilities / total_assets
                    else:
                        metrics['debt_to_assets'] = 0
                        
                    if total_equity != 0:
                        metrics['debt_to_equity'] = total_liabilities / total_equity
                    else:
                        metrics['debt_to_equity'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating solvency ratios: {e}")
                    metrics['debt_to_assets'] = 0
                    metrics['debt_to_equity'] = 0
            
            # Profitability ratios
            if not income_stmt.empty and not balance_sheet.empty:
                try:
                    net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
                    revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
                    
                    if revenue != 0:
                        metrics['profit_margin'] = net_income / revenue
                    else:
                        metrics['profit_margin'] = 0
                        
                    if total_assets != 0:
                        metrics['return_on_assets'] = net_income / total_assets
                    else:
                        metrics['return_on_assets'] = 0
                        
                    if total_equity != 0:
                        metrics['return_on_equity'] = net_income / total_equity
                    else:
                        metrics['return_on_equity'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating profitability ratios: {e}")
                    metrics['profit_margin'] = 0
                    metrics['return_on_assets'] = 0
                    metrics['return_on_equity'] = 0
            
            # Efficiency ratios
            if not income_stmt.empty and not balance_sheet.empty:
                try:
                    if total_assets != 0:
                        metrics['asset_turnover'] = revenue / total_assets
                    else:
                        metrics['asset_turnover'] = 0
                        
                    inventory = balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else 0
                    cogs = income_stmt.loc['Cost Of Revenue'].iloc[0] if 'Cost Of Revenue' in income_stmt.index else 0
                    
                    if inventory != 0:
                        metrics['inventory_turnover'] = cogs / inventory
                    else:
                        metrics['inventory_turnover'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating efficiency ratios: {e}")
                    metrics['asset_turnover'] = 0
                    metrics['inventory_turnover'] = 0
            
            # Growth metrics
            if len(income_stmt.columns) >= 2 and 'Total Revenue' in income_stmt.index:
                try:
                    current_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                    previous_revenue = income_stmt.loc['Total Revenue'].iloc[1]
                    
                    if previous_revenue != 0:
                        metrics['revenue_growth'] = (current_revenue - previous_revenue) / previous_revenue
                    else:
                        metrics['revenue_growth'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating revenue growth: {e}")
                    metrics['revenue_growth'] = 0
            
            if len(income_stmt.columns) >= 2 and 'Net Income' in income_stmt.index:
                try:
                    current_income = income_stmt.loc['Net Income'].iloc[0]
                    previous_income = income_stmt.loc['Net Income'].iloc[1]
                    
                    if previous_income != 0:
                        metrics['income_growth'] = (current_income - previous_income) / previous_income
                    else:
                        metrics['income_growth'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating income growth: {e}")
                    metrics['income_growth'] = 0
            
            # Valuation metrics
            info = stock.info
            metrics['pe_ratio'] = info.get('trailingPE', 0)
            metrics['forward_pe'] = info.get('forwardPE', 0)
            metrics['peg_ratio'] = info.get('pegRatio', 0)
            metrics['price_to_book'] = info.get('priceToBook', 0)
            metrics['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
            metrics['dividend_yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            # Cash flow metrics
            if not cash_flow.empty:
                try:
                    operating_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else 0
                    capital_expenditures = cash_flow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cash_flow.index else 0
                    
                    metrics['free_cash_flow'] = operating_cash_flow + capital_expenditures  # Capital expenditures are negative
                    
                    if revenue != 0:
                        metrics['cash_flow_to_sales'] = operating_cash_flow / revenue
                    else:
                        metrics['cash_flow_to_sales'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating cash flow metrics: {e}")
                    metrics['free_cash_flow'] = 0
                    metrics['cash_flow_to_sales'] = 0
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting financial health metrics for {ticker}: {e}")
            return {}
    
    def get_earnings_analysis(self, ticker):
        """Analyze recent earnings reports and surprises."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get earnings data
            earnings = stock.earnings
            earnings_dates = stock.earnings_dates
            
            # Prepare results
            analysis = {
                'annual_earnings': [],
                'quarterly_earnings': [],
                'earnings_surprises': [],
                'next_earnings_date': None,
                'earnings_trend': 'neutral',
                'surprise_trend': 'neutral'
            }
            
            # Process annual earnings
            if not earnings.empty:
                for year, row in earnings.iterrows():
                    analysis['annual_earnings'].append({
                        'year': int(year),
                        'revenue': float(row['Revenue']),
                        'earnings': float(row['Earnings'])
                    })
            
            # Process quarterly earnings and surprises
            if not earnings_dates.empty:
                surprises = []
                
                for date, row in earnings_dates.iterrows():
                    quarter_data = {
                        'date': date.strftime('%Y-%m-%d'),
                        'reported_eps': float(row['Reported EPS']) if not pd.isna(row['Reported EPS']) else None,
                        'estimated_eps': float(row['Estimated EPS']) if not pd.isna(row['Estimated EPS']) else None,
                        'surprise': float(row['Surprise(%)']) if not pd.isna(row['Surprise(%)']) else None
                    }
                    
                    analysis['quarterly_earnings'].append(quarter_data)
                    
                    if quarter_data['surprise'] is not None:
                        surprises.append(quarter_data['surprise'])
                        analysis['earnings_surprises'].append({
                            'date': quarter_data['date'],
                            'surprise_percent': quarter_data['surprise']
                        })
                
                # Determine earnings trend
                if len(analysis['quarterly_earnings']) >= 2:
                    latest = analysis['quarterly_earnings'][0]['reported_eps']
                    previous = analysis['quarterly_earnings'][1]['reported_eps']
                    
                    if latest is not None and previous is not None:
                        if latest > previous * 1.05:  # 5% growth
                            analysis['earnings_trend'] = 'strong_growth'
                        elif latest > previous:
                            analysis['earnings_trend'] = 'growth'
                        elif latest < previous * 0.95:  # 5% decline
                            analysis['earnings_trend'] = 'strong_decline'
                        elif latest < previous:
                            analysis['earnings_trend'] = 'decline'
                        else:
                            analysis['earnings_trend'] = 'stable'
                
                # Determine surprise trend
                if len(surprises) >= 3:
                    avg_surprise = sum(surprises[:3]) / 3
                    
                    if avg_surprise > 10:
                        analysis['surprise_trend'] = 'strong_positive'
                    elif avg_surprise > 0:
                        analysis['surprise_trend'] = 'positive'
                    elif avg_surprise < -10:
                        analysis['surprise_trend'] = 'strong_negative'
                    elif avg_surprise < 0:
                        analysis['surprise_trend'] = 'negative'
                    else:
                        analysis['surprise_trend'] = 'neutral'
                
                # Get next earnings date
                next_date = stock.calendar
                if not next_date.empty:
                    analysis['next_earnings_date'] = next_date.iloc[0, 0].strftime('%Y-%m-%d') if pd.notna(next_date.iloc[0, 0]) else None
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error getting earnings analysis for {ticker}: {e}")
            return {
                'annual_earnings': [],
                'quarterly_earnings': [],
                'earnings_surprises': [],
                'next_earnings_date': None,
                'earnings_trend': 'neutral',
                'surprise_trend': 'neutral'
            }
    
    def get_analyst_recommendations(self, ticker):
        """Get analyst recommendations and price targets."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get recommendations
            recommendations = stock.recommendations
            
            # Get analyst price targets
            price_targets = stock.analyst_price_target
            
            # Prepare results
            analysis = {
                'recommendations': [],
                'current_consensus': 'neutral',
                'recommendation_trend': 'stable',
                'price_targets': {
                    'low': None,
                    'mean': None,
                    'high': None,
                    'current_price': None,
                    'potential_upside': None
                }
            }
            
            # Process recommendations
            if recommendations is not None and not recommendations.empty:
                for date, row in recommendations.iterrows():
                    analysis['recommendations'].append({
                        'date': date.strftime('%Y-%m-%d'),
                        'firm': row['Firm'] if 'Firm' in row else 'Unknown',
                        'to_grade': row['To Grade'] if 'To Grade' in row else None,
                        'from_grade': row['From Grade'] if 'From Grade' in row else None,
                        'action': row['Action'] if 'Action' in row else None
                    })
                
                # Determine current consensus
                if len(analysis['recommendations']) > 0:
                    latest_grades = [r['to_grade'] for r in analysis['recommendations'][:5] if r['to_grade'] is not None]
                    
                    buy_equivalent = ['buy', 'strong buy', 'outperform', 'overweight', 'positive']
                    sell_equivalent = ['sell', 'strong sell', 'underperform', 'underweight', 'negative']
                    
                    buy_count = sum(1 for grade in latest_grades if grade.lower() in buy_equivalent)
                    sell_count = sum(1 for grade in latest_grades if grade.lower() in sell_equivalent)
                    hold_count = len(latest_grades) - buy_count - sell_count
                    
                    if buy_count > sell_count + hold_count:
                        analysis['current_consensus'] = 'strong_buy'
                    elif buy_count > sell_count:
                        analysis['current_consensus'] = 'buy'
                    elif sell_count > buy_count + hold_count:
                        analysis['current_consensus'] = 'strong_sell'
                    elif sell_count > buy_count:
                        analysis['current_consensus'] = 'sell'
                    else:
                        analysis['current_consensus'] = 'hold'
                
                # Determine recommendation trend
                if len(analysis['recommendations']) >= 10:
                    recent_actions = [r['action'] for r in analysis['recommendations'][:5] if r['action'] is not None]
                    older_actions = [r['action'] for r in analysis['recommendations'][5:10] if r['action'] is not None]
                    
                    upgrade_equivalent = ['upgrade', 'initiated', 'reiterated']
                    downgrade_equivalent = ['downgrade']
                    
                    recent_upgrades = sum(1 for action in recent_actions if action.lower() in upgrade_equivalent)
                    recent_downgrades = sum(1 for action in recent_actions if action.lower() in downgrade_equivalent)
                    older_upgrades = sum(1 for action in older_actions if action.lower() in upgrade_equivalent)
                    older_downgrades = sum(1 for action in older_actions if action.lower() in downgrade_equivalent)
                    
                    recent_sentiment = recent_upgrades - recent_downgrades
                    older_sentiment = older_upgrades - older_downgrades
                    
                    if recent_sentiment > 2 and recent_sentiment > older_sentiment:
                        analysis['recommendation_trend'] = 'improving_strongly'
                    elif recent_sentiment > 0 and recent_sentiment > older_sentiment:
                        analysis['recommendation_trend'] = 'improving'
                    elif recent_sentiment < -2 and recent_sentiment < older_sentiment:
                        analysis['recommendation_trend'] = 'deteriorating_strongly'
                    elif recent_sentiment < 0 and recent_sentiment < older_sentiment:
                        analysis['recommendation_trend'] = 'deteriorating'
                    else:
                        analysis['recommendation_trend'] = 'stable'
            
            # Process price targets
            if price_targets is not None and not price_targets.empty:
                analysis['price_targets']['low'] = float(price_targets.iloc[0, 0]) if pd.notna(price_targets.iloc[0, 0]) else None
                analysis['price_targets']['mean'] = float(price_targets.iloc[0, 1]) if pd.notna(price_targets.iloc[0, 1]) else None
                analysis['price_targets']['high'] = float(price_targets.iloc[0, 2]) if pd.notna(price_targets.iloc[0, 2]) else None
                analysis['price_targets']['current_price'] = float(price_targets.iloc[0, 3]) if pd.notna(price_targets.iloc[0, 3]) else None
                
                if analysis['price_targets']['mean'] is not None and analysis['price_targets']['current_price'] is not None and analysis['price_targets']['current_price'] > 0:
                    analysis['price_targets']['potential_upside'] = (analysis['price_targets']['mean'] - analysis['price_targets']['current_price']) / analysis['price_targets']['current_price'] * 100
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error getting analyst recommendations for {ticker}: {e}")
            return {
                'recommendations': [],
                'current_consensus': 'neutral',
                'recommendation_trend': 'stable',
                'price_targets': {
                    'low': None,
                    'mean': None,
                    'high': None,
                    'current_price': None,
                    'potential_upside': None
                }
            }
    
    def get_news_sentiment_detailed(self, ticker, days=30):
        """Get detailed news sentiment analysis with topic extraction."""
        try:
            # Get news from Yahoo Finance
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                logger.warning(f"No news found for {ticker}")
                return {
                    'overall_sentiment': 0,
                    'sentiment_trend': 'neutral',
                    'news_volume': 0,
                    'sentiment_by_source': {},
                    'sentiment_by_topic': {},
                    'top_headlines': [],
                    'topics': []
                }
            
            # Process news items
            news_items = []
            sources = {}
            topics = {}
            
            for item in news:
                title = item.get('title', '')
                publisher = item.get('publisher', '')
                publish_time = item.get('providerPublishTime', 0)
                
                if title and publish_time:
                    # Calculate sentiment
                    vader_sentiment = self.sentiment_analyzer.polarity_scores(title)
                    textblob_sentiment = TextBlob(title).sentiment
                    
                    # Combined sentiment (average of VADER and TextBlob)
                    compound_sentiment = (vader_sentiment['compound'] + textblob_sentiment.polarity) / 2
                    
                    # Extract topics
                    extracted_topics = self.extract_topics(title)
                    
                    news_item = {
                        'title': title,
                        'publisher': publisher,
                        'date': datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d'),
                        'sentiment': compound_sentiment,
                        'topics': extracted_topics
                    }
                    
                    news_items.append(news_item)
                    
                    # Aggregate by source
                    if publisher:
                        if publisher not in sources:
                            sources[publisher] = {
                                'count': 0,
                                'sentiment_sum': 0
                            }
                        sources[publisher]['count'] += 1
                        sources[publisher]['sentiment_sum'] += compound_sentiment
                    
                    # Aggregate by topic
                    for topic in extracted_topics:
                        if topic not in topics:
                            topics[topic] = {
                                'count': 0,
                                'sentiment_sum': 0
                            }
                        topics[topic]['count'] += 1
                        topics[topic]['sentiment_sum'] += compound_sentiment
            
            # Calculate overall sentiment
            if news_items:
                overall_sentiment = sum(item['sentiment'] for item in news_items) / len(news_items)
            else:
                overall_sentiment = 0
            
            # Determine sentiment trend
            if overall_sentiment > 0.3:
                sentiment_trend = 'very_positive'
            elif overall_sentiment > 0.1:
                sentiment_trend = 'positive'
            elif overall_sentiment > -0.1:
                sentiment_trend = 'neutral'
            elif overall_sentiment > -0.3:
                sentiment_trend = 'negative'
            else:
                sentiment_trend = 'very_negative'
            
            # Calculate sentiment by source
            sentiment_by_source = {}
            for source, data in sources.items():
                if data['count'] > 0:
                    sentiment_by_source[source] = data['sentiment_sum'] / data['count']
            
            # Calculate sentiment by topic
            sentiment_by_topic = {}
            for topic, data in topics.items():
                if data['count'] > 0:
                    sentiment_by_topic[topic] = data['sentiment_sum'] / data['count']
            
            # Sort news items by sentiment impact (absolute value)
            news_items.sort(key=lambda x: abs(x['sentiment']), reverse=True)
            
            # Extract top topics
            topic_list = [{'topic': topic, 'count': data['count'], 'sentiment': data['sentiment_sum'] / data['count']} 
                         for topic, data in topics.items() if data['count'] > 0]
            topic_list.sort(key=lambda x: x['count'], reverse=True)
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_trend': sentiment_trend,
                'news_volume': len(news_items),
                'sentiment_by_source': sentiment_by_source,
                'sentiment_by_topic': sentiment_by_topic,
                'top_headlines': news_items[:10],
                'topics': topic_list[:5]
            }
        
        except Exception as e:
            logger.error(f"Error getting detailed news sentiment for {ticker}: {e}")
            return {
                'overall_sentiment': 0,
                'sentiment_trend': 'neutral',
                'news_volume': 0,
                'sentiment_by_source': {},
                'sentiment_by_topic': {},
                'top_headlines': [],
                'topics': []
            }
    
    def extract_topics(self, text):
        """Extract topics from news headlines."""
        # List of common financial topics
        financial_topics = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline', 'dividend',
            'acquisition', 'merger', 'takeover', 'buyout', 'partnership', 'collaboration',
            'product', 'launch', 'release', 'innovation', 'patent', 'research',
            'lawsuit', 'litigation', 'settlement', 'regulation', 'compliance',
            'ceo', 'executive', 'management', 'leadership', 'board', 'director',
            'stock', 'share', 'price', 'valuation', 'market', 'investor', 'analyst',
            'upgrade', 'downgrade', 'rating', 'recommendation', 'target',
            'debt', 'financing', 'funding', 'investment', 'capital', 'ipo',
            'restructuring', 'layoff', 'cost-cutting', 'efficiency', 'expansion',
            'sales', 'customer', 'user', 'subscriber', 'client', 'contract',
            'forecast', 'guidance', 'outlook', 'projection', 'estimate',
            'economy', 'inflation', 'interest rate', 'fed', 'recession', 'recovery'
        ]
        
        # Extract topics
        text_lower = text.lower()
        found_topics = []
        
        for topic in financial_topics:
            if topic in text_lower:
                found_topics.append(topic)
        
        # If no predefined topics found, extract nouns as potential topics
        if not found_topics:
            # Simple noun extraction based on capitalization patterns
            potential_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
            found_topics = list(set(potential_nouns))[:3]  # Limit to 3 unique nouns
        
        return found_topics
    
    def get_insider_trading(self, ticker):
        """Analyze insider trading patterns."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get insider transactions
            insider_trades = stock.insider_transactions
            
            if insider_trades is None or insider_trades.empty:
                return {
                    'recent_transactions': [],
                    'net_shares': 0,
                    'buy_sell_ratio': 0,
                    'insider_sentiment': 'neutral',
                    'significant_transactions': []
                }
            
            # Process transactions
            transactions = []
            total_bought = 0
            total_sold = 0
            
            for index, row in insider_trades.iterrows():
                try:
                    transaction = {
                        'date': index.strftime('%Y-%m-%d') if isinstance(index, pd.Timestamp) else str(index),
                        'insider': row['Insider'] if 'Insider' in row else 'Unknown',
                        'title': row['Title'] if 'Title' in row else 'Unknown',
                        'transaction': row['Transaction'] if 'Transaction' in row else 'Unknown',
                        'shares': int(row['Shares']) if 'Shares' in row and pd.notna(row['Shares']) else 0,
                        'value': float(row['Value']) if 'Value' in row and pd.notna(row['Value']) else 0,
                        'shares_total': int(row['Shares Total']) if 'Shares Total' in row and pd.notna(row['Shares Total']) else 0
                    }
                    
                    transactions.append(transaction)
                    
                    # Track buy/sell volumes
                    if 'buy' in transaction['transaction'].lower():
                        total_bought += transaction['shares']
                    elif 'sell' in transaction['transaction'].lower():
                        total_sold += transaction['shares']
                except Exception as e:
                    logger.warning(f"Error processing insider transaction: {e}")
                    continue
            
            # Calculate metrics
            net_shares = total_bought - total_sold
            buy_sell_ratio = total_bought / total_sold if total_sold > 0 else float('inf')
            
            # Determine insider sentiment
            if buy_sell_ratio > 3:
                insider_sentiment = 'very_bullish'
            elif buy_sell_ratio > 1.5:
                insider_sentiment = 'bullish'
            elif buy_sell_ratio > 0.67:  # 2/3
                insider_sentiment = 'neutral'
            elif buy_sell_ratio > 0.33:  # 1/3
                insider_sentiment = 'bearish'
            else:
                insider_sentiment = 'very_bearish'
            
            # Identify significant transactions (large volume or by key executives)
            significant_transactions = []
            key_titles = ['ceo', 'chief', 'president', 'director', 'chairman']
            
            for transaction in transactions:
                is_key_executive = any(title.lower() in transaction['title'].lower() for title in key_titles)
                is_large_transaction = transaction['shares'] > 10000 or transaction['value'] > 1000000
                
                if is_key_executive or is_large_transaction:
                    significant_transactions.append(transaction)
            
            return {
                'recent_transactions': transactions[:10],
                'net_shares': net_shares,
                'buy_sell_ratio': buy_sell_ratio,
                'insider_sentiment': insider_sentiment,
                'significant_transactions': significant_transactions[:5]
            }
        
        except Exception as e:
            logger.error(f"Error getting insider trading for {ticker}: {e}")
            return {
                'recent_transactions': [],
                'net_shares': 0,
                'buy_sell_ratio': 0,
                'insider_sentiment': 'neutral',
                'significant_transactions': []
            }
    
    def get_institutional_ownership(self, ticker):
        """Analyze institutional ownership and changes."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get institutional holders
            institutional_holders = stock.institutional_holders
            
            # Get major holders
            major_holders = stock.major_holders
            
            # Prepare results
            analysis = {
                'top_institutions': [],
                'total_institutional_ownership': 0,
                'institutional_count': 0,
                'ownership_concentration': 0,
                'recent_changes': []
            }
            
            # Process institutional holders
            if institutional_holders is not None and not institutional_holders.empty:
                total_shares_held = 0
                
                for index, row in institutional_holders.iterrows():
                    institution = {
                        'name': row['Holder'] if 'Holder' in row else 'Unknown',
                        'shares': int(row['Shares']) if 'Shares' in row and pd.notna(row['Shares']) else 0,
                        'date_reported': row['Date Reported'].strftime('%Y-%m-%d') if 'Date Reported' in row and isinstance(row['Date Reported'], pd.Timestamp) else None,
                        'percent_out': float(row['% Out']) if '% Out' in row and pd.notna(row['% Out']) else 0,
                        'value': float(row['Value']) if 'Value' in row and pd.notna(row['Value']) else 0
                    }
                    
                    analysis['top_institutions'].append(institution)
                    total_shares_held += institution['shares']
                
                analysis['institutional_count'] = len(institutional_holders)
                
                # Calculate concentration (% held by top 5)
                if analysis['institutional_count'] > 0:
                    top5_shares = sum(inst['shares'] for inst in analysis['top_institutions'][:5])
                    analysis['ownership_concentration'] = top5_shares / total_shares_held if total_shares_held > 0 else 0
            
            # Get total institutional ownership from major holders
            if major_holders is not None and not major_holders.empty and len(major_holders) >= 2:
                try:
                    # Major holders typically has institutional ownership in the first row
                    inst_ownership_str = major_holders.iloc[0, 0]
                    if isinstance(inst_ownership_str, str) and '%' in inst_ownership_str:
                        analysis['total_institutional_ownership'] = float(inst_ownership_str.strip('%')) / 100
                except Exception as e:
                    logger.warning(f"Error extracting institutional ownership percentage: {e}")
            
            # Get recent changes (this is a placeholder - Yahoo Finance API doesn't provide historical changes)
            # In a real implementation, you would track changes over time or use a different data source
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error getting institutional ownership for {ticker}: {e}")
            return {
                'top_institutions': [],
                'total_institutional_ownership': 0,
                'institutional_count': 0,
                'ownership_concentration': 0,
                'recent_changes': []
            }
    
    def get_macro_economic_factors(self):
        """Get current macroeconomic factors that might impact the market."""
        try:
            # This would typically come from an economic data API
            # For demonstration, we'll use placeholder data
            
            # In a real implementation, you would fetch this data from sources like:
            # - Federal Reserve Economic Data (FRED)
            # - Bureau of Economic Analysis
            # - Bureau of Labor Statistics
            # - World Bank or IMF data
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            macro_data = {
                'date': current_date,
                'indicators': {
                    'gdp_growth': 2.1,  # Annual GDP growth rate (%)
                    'inflation_rate': 3.2,  # Annual inflation rate (%)
                    'unemployment_rate': 3.8,  # Unemployment rate (%)
                    'fed_funds_rate': 5.25,  # Federal funds rate (%)
                    'treasury_10y_yield': 4.2,  # 10-year Treasury yield (%)
                    'treasury_2y_yield': 4.8,  # 2-year Treasury yield (%)
                    'yield_curve': -0.6,  # 10y-2y spread (%)
                    'dollar_index': 104.5,  # US Dollar Index
                    'vix': 18.5,  # CBOE Volatility Index
                    'consumer_sentiment': 79.2,  # University of Michigan Consumer Sentiment
                    'ism_manufacturing': 48.5,  # ISM Manufacturing PMI
                    'ism_services': 52.3,  # ISM Services PMI
                    'retail_sales_growth': 1.8,  # Monthly retail sales growth (%)
                    'housing_starts': 1.45,  # Housing starts (millions)
                    'oil_price': 82.5  # WTI Crude Oil price (USD)
                },
                'trends': {
                    'gdp_trend': 'stable',
                    'inflation_trend': 'decreasing',
                    'rate_trend': 'stable',
                    'yield_curve_trend': 'inverting',
                    'dollar_trend': 'strengthening',
                    'consumer_trend': 'weakening',
                    'manufacturing_trend': 'contracting',
                    'services_trend': 'expanding',
                    'housing_trend': 'weakening',
                    'energy_trend': 'volatile'
                },
                'market_phase': 'late_cycle',
                'recession_probability': 35,  # Probability of recession in next 12 months (%)
                'fed_policy_bias': 'neutral',  # Current Federal Reserve policy bias
                'global_risks': [
                    'geopolitical tensions',
                    'trade disputes',
                    'supply chain disruptions',
                    'energy price volatility',
                    'climate change impacts'
                ]
            }
            
            return macro_data
        
        except Exception as e:
            logger.error(f"Error getting macroeconomic factors: {e}")
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'indicators': {},
                'trends': {},
                'market_phase': 'unknown',
                'recession_probability': 0,
                'fed_policy_bias': 'unknown',
                'global_risks': []
            }
    
    def get_sector_performance(self, ticker):
        """Get sector performance and relative strength."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get ticker's sector
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Get sector ETFs (placeholder - in a real implementation, you would use actual sector ETF data)
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Materials': 'XLB',
                'Industrials': 'XLI',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC'
            }
            
            # Get S&P 500 for market comparison
            market_etf = 'SPY'
            
            # Find corresponding sector ETF
            sector_etf = sector_etfs.get(sector, market_etf)
            
            # Get historical data for ticker, sector, and market
            ticker_data = yf.download(ticker, period='1y', interval='1d', progress=False)
            sector_data = yf.download(sector_etf, period='1y', interval='1d', progress=False)
            market_data = yf.download(market_etf, period='1y', interval='1d', progress=False)
            
            # Calculate returns
            ticker_returns = ticker_data['Close'].pct_change()
            sector_returns = sector_data['Close'].pct_change()
            market_returns = market_data['Close'].pct_change()
            
            # Calculate performance metrics
            ticker_ytd = (ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[0] - 1) * 100
            sector_ytd = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[0] - 1) * 100
            market_ytd = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[0] - 1) * 100
            
            # Calculate relative strength (vs sector and market)
            rs_vs_sector = ticker_ytd - sector_ytd
            rs_vs_market = ticker_ytd - market_ytd
            
            # Calculate correlation
            correlation_with_sector = ticker_returns.corr(sector_returns)
            correlation_with_market = ticker_returns.corr(market_returns)
            
            # Calculate beta
            covariance_with_market = ticker_returns.cov(market_returns)
            market_variance = market_returns.var()
            beta = covariance_with_market / market_variance if market_variance != 0 else 1
            
            # Determine sector trend
            if sector_ytd > 20:
                sector_trend = 'strong_uptrend'
            elif sector_ytd > 5:
                sector_trend = 'uptrend'
            elif sector_ytd > -5:
                sector_trend = 'neutral'
            elif sector_ytd > -20:
                sector_trend = 'downtrend'
            else:
                sector_trend = 'strong_downtrend'
            
            # Determine relative strength trend
            if rs_vs_sector > 10 and rs_vs_market > 10:
                rs_trend = 'strong_outperformance'
            elif rs_vs_sector > 0 and rs_vs_market > 0:
                rs_trend = 'outperformance'
            elif rs_vs_sector < -10 and rs_vs_market < -10:
                rs_trend = 'strong_underperformance'
            elif rs_vs_sector < 0 and rs_vs_market < 0:
                rs_trend = 'underperformance'
            else:
                rs_trend = 'mixed'
            
            # Create sector performance chart
            plt.figure(figsize=(10, 6))
            
            # Normalize prices to 100 at the start
            ticker_norm = ticker_data['Close'] / ticker_data['Close'].iloc[0] * 100
            sector_norm = sector_data['Close'] / sector_data['Close'].iloc[0] * 100
            market_norm = market_data['Close'] / market_data['Close'].iloc[0] * 100
            
            plt.plot(ticker_norm.index, ticker_norm, label=ticker)
            plt.plot(sector_norm.index, sector_norm, label=f"{sector} ({sector_etf})")
            plt.plot(market_norm.index, market_norm, label=f"S&P 500 ({market_etf})")
            
            plt.title(f"Relative Performance: {ticker} vs {sector} vs S&P 500")
            plt.xlabel("Date")
            plt.ylabel("Normalized Price (Base 100)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save chart to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64 for embedding
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return {
                'sector': sector,
                'industry': industry,
                'ticker_ytd_return': ticker_ytd,
                'sector_ytd_return': sector_ytd,
                'market_ytd_return': market_ytd,
                'relative_strength_vs_sector': rs_vs_sector,
                'relative_strength_vs_market': rs_vs_market,
                'correlation_with_sector': correlation_with_sector,
                'correlation_with_market': correlation_with_market,
                'beta': beta,
                'sector_trend': sector_trend,
                'relative_strength_trend': rs_trend,
                'sector_etf': sector_etf,
                'performance_chart': chart_base64
            }
        
        except Exception as e:
            logger.error(f"Error getting sector performance for {ticker}: {e}")
            return {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'ticker_ytd_return': 0,
                'sector_ytd_return': 0,
                'market_ytd_return': 0,
                'relative_strength_vs_sector': 0,
                'relative_strength_vs_market': 0,
                'correlation_with_sector': 0,
                'correlation_with_market': 0,
                'beta': 1,
                'sector_trend': 'unknown',
                'relative_strength_trend': 'unknown',
                'sector_etf': 'SPY',
                'performance_chart': None
            }
    
    def generate_qualitative_analysis(self, ticker):
        """Generate comprehensive qualitative analysis for a ticker."""
        try:
            # Get company profile
            profile = self.get_company_profile(ticker)
            
            # Get financial health metrics
            financial_health = self.get_financial_health_metrics(ticker)
            
            # Get earnings analysis
            earnings = self.get_earnings_analysis(ticker)
            
            # Get analyst recommendations
            analyst_recs = self.get_analyst_recommendations(ticker)
            
            # Get detailed news sentiment
            news_sentiment = self.get_news_sentiment_detailed(ticker)
            
            # Get insider trading
            insider_trading = self.get_insider_trading(ticker)
            
            # Get institutional ownership
            institutional = self.get_institutional_ownership(ticker)
            
            # Get macroeconomic factors
            macro = self.get_macro_economic_factors()
            
            # Get sector performance
            sector = self.get_sector_performance(ticker)
            
            # Generate narrative summary
            narrative = self.generate_narrative_summary(
                ticker, profile, financial_health, earnings, analyst_recs,
                news_sentiment, insider_trading, institutional, macro, sector
            )
            
            # Compile complete analysis
            analysis = {
                'ticker': ticker,
                'date_generated': datetime.now().strftime('%Y-%m-%d'),
                'company_profile': profile,
                'financial_health': financial_health,
                'earnings_analysis': earnings,
                'analyst_recommendations': analyst_recs,
                'news_sentiment': news_sentiment,
                'insider_trading': insider_trading,
                'institutional_ownership': institutional,
                'macroeconomic_factors': macro,
                'sector_performance': sector,
                'narrative_summary': narrative
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error generating qualitative analysis for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_narrative_summary(self, ticker, profile, financial_health, earnings, analyst_recs,
                                  news_sentiment, insider_trading, institutional, macro, sector):
        """Generate a narrative summary of the qualitative analysis."""
        try:
            # Company overview
            company_overview = f"{profile['name']} ({ticker}) is a {profile['sector']} company in the {profile['industry']} industry. "
            
            if profile['business_summary'] and len(profile['business_summary']) > 20:
                # Truncate business summary if too long
                summary = profile['business_summary']
                if len(summary) > 300:
                    summary = summary[:300] + "..."
                company_overview += f"{summary} "
            
            company_overview += f"The company has a market capitalization of ${profile['market_cap']/1e9:.2f} billion. "
            
            # Financial health
            financial_narrative = "From a financial health perspective, "
            
            if financial_health:
                if 'current_ratio' in financial_health and financial_health['current_ratio'] > 0:
                    if financial_health['current_ratio'] > 2:
                        financial_narrative += f"the company has a strong liquidity position with a current ratio of {financial_health['current_ratio']:.2f}. "
                    elif financial_health['current_ratio'] > 1:
                        financial_narrative += f"the company has an adequate liquidity position with a current ratio of {financial_health['current_ratio']:.2f}. "
                    else:
                        financial_narrative += f"the company has potential liquidity concerns with a current ratio of {financial_health['current_ratio']:.2f}. "
                
                if 'debt_to_equity' in financial_health:
                    if financial_health['debt_to_equity'] < 0.5:
                        financial_narrative += "The company has a conservative capital structure with low leverage. "
                    elif financial_health['debt_to_equity'] < 1.5:
                        financial_narrative += "The company has a moderate level of financial leverage. "
                    else:
                        financial_narrative += "The company has a high level of financial leverage, which may increase financial risk. "
                
                if 'profit_margin' in financial_health:
                    if financial_health['profit_margin'] > 0.2:
                        financial_narrative += f"Profitability is strong with a {financial_health['profit_margin']*100:.1f}% profit margin. "
                    elif financial_health['profit_margin'] > 0.05:
                        financial_narrative += f"Profitability is moderate with a {financial_health['profit_margin']*100:.1f}% profit margin. "
                    elif financial_health['profit_margin'] > 0:
                        financial_narrative += f"Profitability is slim with only a {financial_health['profit_margin']*100:.1f}% profit margin. "
                    else:
                        financial_narrative += "The company is currently unprofitable. "
            else:
                financial_narrative += "detailed financial metrics are not available. "
            
            # Earnings trend
            earnings_narrative = "Regarding earnings performance, "
            
            if earnings['earnings_trend'] == 'strong_growth':
                earnings_narrative += "the company has shown strong earnings growth in recent quarters. "
            elif earnings['earnings_trend'] == 'growth':
                earnings_narrative += "the company has shown positive earnings growth in recent quarters. "
            elif earnings['earnings_trend'] == 'stable':
                earnings_narrative += "the company's earnings have been stable in recent quarters. "
            elif earnings['earnings_trend'] == 'decline':
                earnings_narrative += "the company has shown some earnings decline in recent quarters. "
            elif earnings['earnings_trend'] == 'strong_decline':
                earnings_narrative += "the company has shown significant earnings decline in recent quarters. "
            else:
                earnings_narrative += "the earnings trend is unclear based on recent quarters. "
            
            if earnings['surprise_trend'] == 'strong_positive':
                earnings_narrative += "The company has consistently exceeded analyst expectations by a significant margin. "
            elif earnings['surprise_trend'] == 'positive':
                earnings_narrative += "The company has generally exceeded analyst expectations. "
            elif earnings['surprise_trend'] == 'negative' or earnings['surprise_trend'] == 'strong_negative':
                earnings_narrative += "The company has fallen short of analyst expectations. "
            
            if earnings['next_earnings_date']:
                earnings_narrative += f"The next earnings report is expected around {earnings['next_earnings_date']}. "
            
            # Analyst sentiment
            analyst_narrative = "Wall Street analysts "
            
            if analyst_recs['current_consensus'] == 'strong_buy':
                analyst_narrative += "are very bullish on the stock with a strong buy consensus. "
            elif analyst_recs['current_consensus'] == 'buy':
                analyst_narrative += "are generally positive on the stock with a buy consensus. "
            elif analyst_recs['current_consensus'] == 'hold':
                analyst_narrative += "have a neutral stance on the stock with a hold consensus. "
            elif analyst_recs['current_consensus'] == 'sell':
                analyst_narrative += "are generally negative on the stock with a sell consensus. "
            elif analyst_recs['current_consensus'] == 'strong_sell':
                analyst_narrative += "are very bearish on the stock with a strong sell consensus. "
            
            if analyst_recs['recommendation_trend'] == 'improving_strongly' or analyst_recs['recommendation_trend'] == 'improving':
                analyst_narrative += "Analyst sentiment has been improving recently. "
            elif analyst_recs['recommendation_trend'] == 'deteriorating_strongly' or analyst_recs['recommendation_trend'] == 'deteriorating':
                analyst_narrative += "Analyst sentiment has been deteriorating recently. "
            
            if analyst_recs['price_targets']['mean'] and analyst_recs['price_targets']['current_price']:
                if analyst_recs['price_targets']['potential_upside'] > 15:
                    analyst_narrative += f"The average price target of ${analyst_recs['price_targets']['mean']:.2f} implies significant upside potential of {analyst_recs['price_targets']['potential_upside']:.1f}%. "
                elif analyst_recs['price_targets']['potential_upside'] > 0:
                    analyst_narrative += f"The average price target of ${analyst_recs['price_targets']['mean']:.2f} implies modest upside potential of {analyst_recs['price_targets']['potential_upside']:.1f}%. "
                elif analyst_recs['price_targets']['potential_upside'] > -15:
                    analyst_narrative += f"The average price target of ${analyst_recs['price_targets']['mean']:.2f} implies a potential downside of {-analyst_recs['price_targets']['potential_upside']:.1f}%. "
                else:
                    analyst_narrative += f"The average price target of ${analyst_recs['price_targets']['mean']:.2f} implies significant downside risk of {-analyst_recs['price_targets']['potential_upside']:.1f}%. "
            
            # News sentiment
            news_narrative = "Recent news sentiment "
            
            if news_sentiment['sentiment_trend'] == 'very_positive':
                news_narrative += f"has been very positive across {news_sentiment['news_volume']} news items. "
            elif news_sentiment['sentiment_trend'] == 'positive':
                news_narrative += f"has been positive across {news_sentiment['news_volume']} news items. "
            elif news_sentiment['sentiment_trend'] == 'neutral':
                news_narrative += f"has been neutral across {news_sentiment['news_volume']} news items. "
            elif news_sentiment['sentiment_trend'] == 'negative':
                news_narrative += f"has been negative across {news_sentiment['news_volume']} news items. "
            elif news_sentiment['sentiment_trend'] == 'very_negative':
                news_narrative += f"has been very negative across {news_sentiment['news_volume']} news items. "
            
            if news_sentiment['topics'] and len(news_sentiment['topics']) > 0:
                top_topics = [topic['topic'] for topic in news_sentiment['topics'][:3]]
                news_narrative += f"Key topics in recent news include {', '.join(top_topics)}. "
            
            if news_sentiment['top_headlines'] and len(news_sentiment['top_headlines']) > 0:
                news_narrative += f"A notable recent headline: \"{news_sentiment['top_headlines'][0]['title']}\". "
            
            # Insider activity
            insider_narrative = "Insider trading activity "
            
            if insider_trading['insider_sentiment'] == 'very_bullish':
                insider_narrative += "has been very bullish with significant insider buying. "
            elif insider_trading['insider_sentiment'] == 'bullish':
                insider_narrative += "has been bullish with more insider buying than selling. "
            elif insider_trading['insider_sentiment'] == 'neutral':
                insider_narrative += "has been relatively balanced between buying and selling. "
            elif insider_trading['insider_sentiment'] == 'bearish':
                insider_narrative += "has been bearish with more insider selling than buying. "
            elif insider_trading['insider_sentiment'] == 'very_bearish':
                insider_narrative += "has been very bearish with significant insider selling. "
            
            if insider_trading['significant_transactions'] and len(insider_trading['significant_transactions']) > 0:
                top_transaction = insider_trading['significant_transactions'][0]
                insider_narrative += f"A notable recent transaction: {top_transaction['insider']} ({top_transaction['title']}) {top_transaction['transaction'].lower()} {top_transaction['shares']} shares. "
            
            # Institutional ownership
            institutional_narrative = "Institutional investors "
            
            if institutional['total_institutional_ownership'] > 0.8:
                institutional_narrative += f"hold a very high {institutional['total_institutional_ownership']*100:.1f}% of outstanding shares. "
            elif institutional['total_institutional_ownership'] > 0.6:
                institutional_narrative += f"hold a high {institutional['total_institutional_ownership']*100:.1f}% of outstanding shares. "
            elif institutional['total_institutional_ownership'] > 0.4:
                institutional_narrative += f"hold a moderate {institutional['total_institutional_ownership']*100:.1f}% of outstanding shares. "
            elif institutional['total_institutional_ownership'] > 0:
                institutional_narrative += f"hold a relatively low {institutional['total_institutional_ownership']*100:.1f}% of outstanding shares. "
            else:
                institutional_narrative += "ownership data is not available. "
            
            if institutional['ownership_concentration'] > 0.6:
                institutional_narrative += "Ownership is highly concentrated among the top institutions. "
            elif institutional['ownership_concentration'] > 0.4:
                institutional_narrative += "Ownership is moderately concentrated among the top institutions. "
            elif institutional['ownership_concentration'] > 0:
                institutional_narrative += "Ownership is well distributed among various institutions. "
            
            # Sector performance
            sector_narrative = f"In terms of sector performance, {ticker} is in the {sector['sector']} sector, which "
            
            if sector['sector_trend'] == 'strong_uptrend':
                sector_narrative += f"has been in a strong uptrend with a {sector['sector_ytd_return']:.1f}% YTD return. "
            elif sector['sector_trend'] == 'uptrend':
                sector_narrative += f"has been in an uptrend with a {sector['sector_ytd_return']:.1f}% YTD return. "
            elif sector['sector_trend'] == 'neutral':
                sector_narrative += f"has been relatively flat with a {sector['sector_ytd_return']:.1f}% YTD return. "
            elif sector['sector_trend'] == 'downtrend':
                sector_narrative += f"has been in a downtrend with a {sector['sector_ytd_return']:.1f}% YTD return. "
            elif sector['sector_trend'] == 'strong_downtrend':
                sector_narrative += f"has been in a strong downtrend with a {sector['sector_ytd_return']:.1f}% YTD return. "
            
            if sector['relative_strength_trend'] == 'strong_outperformance':
                sector_narrative += f"The stock has significantly outperformed both its sector (by {sector['relative_strength_vs_sector']:.1f}%) and the broader market (by {sector['relative_strength_vs_market']:.1f}%). "
            elif sector['relative_strength_trend'] == 'outperformance':
                sector_narrative += f"The stock has outperformed both its sector (by {sector['relative_strength_vs_sector']:.1f}%) and the broader market (by {sector['relative_strength_vs_market']:.1f}%). "
            elif sector['relative_strength_trend'] == 'underperformance':
                sector_narrative += f"The stock has underperformed both its sector (by {-sector['relative_strength_vs_sector']:.1f}%) and the broader market (by {-sector['relative_strength_vs_market']:.1f}%). "
            elif sector['relative_strength_trend'] == 'strong_underperformance':
                sector_narrative += f"The stock has significantly underperformed both its sector (by {-sector['relative_strength_vs_sector']:.1f}%) and the broader market (by {-sector['relative_strength_vs_market']:.1f}%). "
            elif sector['relative_strength_trend'] == 'mixed':
                sector_narrative += "The stock has shown mixed performance relative to its sector and the broader market. "
            
            # Macroeconomic context
            macro_narrative = "From a macroeconomic perspective, "
            
            if macro['market_phase'] == 'early_cycle':
                macro_narrative += "the economy appears to be in an early cycle phase with recovery underway. "
            elif macro['market_phase'] == 'mid_cycle':
                macro_narrative += "the economy appears to be in a mid-cycle expansion phase. "
            elif macro['market_phase'] == 'late_cycle':
                macro_narrative += "the economy appears to be in a late-cycle phase with potential slowdown ahead. "
            elif macro['market_phase'] == 'recession':
                macro_narrative += "the economy appears to be in a recessionary phase. "
            
            if 'fed_funds_rate' in macro['indicators']:
                macro_narrative += f"The Federal Reserve's policy rate stands at {macro['indicators']['fed_funds_rate']}% with a {macro['fed_policy_bias']} bias. "
            
            if 'inflation_rate' in macro['indicators']:
                macro_narrative += f"Inflation is running at {macro['indicators']['inflation_rate']}% and is {macro['trends']['inflation_trend']}. "
            
            if macro['recession_probability'] > 50:
                macro_narrative += f"There is a high probability ({macro['recession_probability']}%) of recession in the next 12 months. "
            elif macro['recession_probability'] > 30:
                macro_narrative += f"There is a moderate probability ({macro['recession_probability']}%) of recession in the next 12 months. "
            else:
                macro_narrative += f"The probability of recession in the next 12 months is relatively low ({macro['recession_probability']}%). "
            
            # Investment conclusion
            conclusion = "In conclusion, "
            
            # Determine overall sentiment based on various factors
            positive_factors = 0
            negative_factors = 0
            
            # Analyst sentiment
            if analyst_recs['current_consensus'] in ['strong_buy', 'buy']:
                positive_factors += 1
            elif analyst_recs['current_consensus'] in ['sell', 'strong_sell']:
                negative_factors += 1
            
            # News sentiment
            if news_sentiment['sentiment_trend'] in ['very_positive', 'positive']:
                positive_factors += 1
            elif news_sentiment['sentiment_trend'] in ['negative', 'very_negative']:
                negative_factors += 1
            
            # Insider sentiment
            if insider_trading['insider_sentiment'] in ['very_bullish', 'bullish']:
                positive_factors += 1
            elif insider_trading['insider_sentiment'] in ['bearish', 'very_bearish']:
                negative_factors += 1
            
            # Earnings trend
            if earnings['earnings_trend'] in ['strong_growth', 'growth']:
                positive_factors += 1
            elif earnings['earnings_trend'] in ['decline', 'strong_decline']:
                negative_factors += 1
            
            # Relative strength
            if sector['relative_strength_trend'] in ['strong_outperformance', 'outperformance']:
                positive_factors += 1
            elif sector['relative_strength_trend'] in ['underperformance', 'strong_underperformance']:
                negative_factors += 1
            
            # Generate conclusion based on balance of factors
            if positive_factors > negative_factors + 1:
                conclusion += f"the qualitative analysis for {ticker} is predominantly positive. "
                conclusion += "The company benefits from "
                
                positive_points = []
                if analyst_recs['current_consensus'] in ['strong_buy', 'buy']:
                    positive_points.append("favorable analyst sentiment")
                if news_sentiment['sentiment_trend'] in ['very_positive', 'positive']:
                    positive_points.append("positive news coverage")
                if insider_trading['insider_sentiment'] in ['very_bullish', 'bullish']:
                    positive_points.append("bullish insider activity")
                if earnings['earnings_trend'] in ['strong_growth', 'growth']:
                    positive_points.append("strong earnings growth")
                if sector['relative_strength_trend'] in ['strong_outperformance', 'outperformance']:
                    positive_points.append("outperformance relative to peers")
                
                conclusion += ", ".join(positive_points[:3]) + ". "
                
                if negative_factors > 0:
                    conclusion += "However, investors should be mindful of "
                    
                    negative_points = []
                    if analyst_recs['current_consensus'] in ['sell', 'strong_sell']:
                        negative_points.append("negative analyst sentiment")
                    if news_sentiment['sentiment_trend'] in ['negative', 'very_negative']:
                        negative_points.append("negative news coverage")
                    if insider_trading['insider_sentiment'] in ['bearish', 'very_bearish']:
                        negative_points.append("bearish insider activity")
                    if earnings['earnings_trend'] in ['decline', 'strong_decline']:
                        negative_points.append("declining earnings")
                    if sector['relative_strength_trend'] in ['underperformance', 'strong_underperformance']:
                        negative_points.append("underperformance relative to peers")
                    
                    conclusion += ", ".join(negative_points) + ". "
            
            elif negative_factors > positive_factors + 1:
                conclusion += f"the qualitative analysis for {ticker} is predominantly negative. "
                conclusion += "The company faces challenges from "
                
                negative_points = []
                if analyst_recs['current_consensus'] in ['sell', 'strong_sell']:
                    negative_points.append("negative analyst sentiment")
                if news_sentiment['sentiment_trend'] in ['negative', 'very_negative']:
                    negative_points.append("negative news coverage")
                if insider_trading['insider_sentiment'] in ['bearish', 'very_bearish']:
                    negative_points.append("bearish insider activity")
                if earnings['earnings_trend'] in ['decline', 'strong_decline']:
                    negative_points.append("declining earnings")
                if sector['relative_strength_trend'] in ['underperformance', 'strong_underperformance']:
                    negative_points.append("underperformance relative to peers")
                
                conclusion += ", ".join(negative_points[:3]) + ". "
                
                if positive_factors > 0:
                    conclusion += "On the positive side, the company does benefit from "
                    
                    positive_points = []
                    if analyst_recs['current_consensus'] in ['strong_buy', 'buy']:
                        positive_points.append("favorable analyst sentiment")
                    if news_sentiment['sentiment_trend'] in ['very_positive', 'positive']:
                        positive_points.append("positive news coverage")
                    if insider_trading['insider_sentiment'] in ['very_bullish', 'bullish']:
                        positive_points.append("bullish insider activity")
                    if earnings['earnings_trend'] in ['strong_growth', 'growth']:
                        positive_points.append("strong earnings growth")
                    if sector['relative_strength_trend'] in ['strong_outperformance', 'outperformance']:
                        positive_points.append("outperformance relative to peers")
                    
                    conclusion += ", ".join(positive_points) + ". "
            
            else:
                conclusion += f"the qualitative analysis for {ticker} is mixed with both positive and negative factors. "
                
                positive_points = []
                if analyst_recs['current_consensus'] in ['strong_buy', 'buy']:
                    positive_points.append("favorable analyst sentiment")
                if news_sentiment['sentiment_trend'] in ['very_positive', 'positive']:
                    positive_points.append("positive news coverage")
                if insider_trading['insider_sentiment'] in ['very_bullish', 'bullish']:
                    positive_points.append("bullish insider activity")
                if earnings['earnings_trend'] in ['strong_growth', 'growth']:
                    positive_points.append("strong earnings growth")
                if sector['relative_strength_trend'] in ['strong_outperformance', 'outperformance']:
                    positive_points.append("outperformance relative to peers")
                
                negative_points = []
                if analyst_recs['current_consensus'] in ['sell', 'strong_sell']:
                    negative_points.append("negative analyst sentiment")
                if news_sentiment['sentiment_trend'] in ['negative', 'very_negative']:
                    negative_points.append("negative news coverage")
                if insider_trading['insider_sentiment'] in ['bearish', 'very_bearish']:
                    negative_points.append("bearish insider activity")
                if earnings['earnings_trend'] in ['decline', 'strong_decline']:
                    negative_points.append("declining earnings")
                if sector['relative_strength_trend'] in ['underperformance', 'strong_underperformance']:
                    negative_points.append("underperformance relative to peers")
                
                if positive_points:
                    conclusion += "Positive factors include " + ", ".join(positive_points[:2]) + ". "
                
                if negative_points:
                    conclusion += "Negative factors include " + ", ".join(negative_points[:2]) + ". "
            
            # Add macroeconomic context to conclusion
            conclusion += "Investors should also consider the current "
            
            if macro['market_phase'] == 'late_cycle':
                conclusion += "late-cycle economic environment, which may present headwinds for certain sectors. "
            elif macro['market_phase'] == 'early_cycle':
                conclusion += "early-cycle economic environment, which typically favors cyclical and growth-oriented investments. "
            elif macro['market_phase'] == 'mid_cycle':
                conclusion += "mid-cycle economic environment, which typically provides a balanced backdrop for various sectors. "
            elif macro['market_phase'] == 'recession':
                conclusion += "recessionary economic environment, which typically favors defensive sectors and quality companies with strong balance sheets. "
            
            # Combine all sections
            full_narrative = (
                company_overview + "\n\n" +
                financial_narrative + "\n\n" +
                earnings_narrative + "\n\n" +
                analyst_narrative + "\n\n" +
                news_narrative + "\n\n" +
                insider_narrative + "\n\n" +
                institutional_narrative + "\n\n" +
                sector_narrative + "\n\n" +
                macro_narrative + "\n\n" +
                conclusion
            )
            
            return full_narrative
        
        except Exception as e:
            logger.error(f"Error generating narrative summary: {e}")
            return f"Unable to generate complete narrative summary due to an error: {e}"

# Example usage
if __name__ == "__main__":
    analyzer = QualitativeAnalyzer()
    analysis = analyzer.generate_qualitative_analysis("AAPL")
    
    if isinstance(analysis, dict) and 'narrative_summary' in analysis:
        print("Qualitative Analysis Summary:")
        print(analysis['narrative_summary'])
    else:
        print(f"Error: {analysis.get('error', 'Unknown error')}")
