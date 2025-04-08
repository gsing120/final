import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForwardLookingStrategy")

class ForwardLookingStrategyGenerator:
    """
    Generates forward-looking trading strategies that predict future market movements
    rather than just analyzing past signals.
    """
    
    def __init__(self):
        """Initialize the forward-looking strategy generator."""
        # Download NLTK resources if not already present
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        
    def get_market_data(self, ticker, period="1y", interval="1d"):
        """Get historical market data for a ticker."""
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                logger.error(f"No data found for {ticker}")
                return None
                
            return data
        
        except Exception as e:
            logger.error(f"Error getting market data for {ticker}: {e}")
            return None
    
    def get_company_fundamentals(self, ticker):
        """Get fundamental data for a company."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get key financial metrics
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Extract key metrics
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 0),
                'eps': info.get('trailingEps', 0),
                'revenue_growth': 0,  # Will calculate if data available
                'profit_margin': info.get('profitMargin', 0) * 100 if info.get('profitMargin') else 0,
                'debt_to_equity': 0,  # Will calculate if data available
                'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            }
            
            # Calculate revenue growth if data available
            if not financials.empty and 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue']
                if len(revenues) >= 2:
                    latest = revenues.iloc[0]
                    previous = revenues.iloc[1]
                    if previous != 0:
                        fundamentals['revenue_growth'] = ((latest - previous) / previous) * 100
            
            # Calculate debt to equity if data available
            if not balance_sheet.empty and 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                debt = balance_sheet.loc['Total Debt'].iloc[0]
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                if equity != 0:
                    fundamentals['debt_to_equity'] = debt / equity
            
            return fundamentals
        
        except Exception as e:
            logger.error(f"Error getting fundamentals for {ticker}: {e}")
            return {}
    
    def get_news_sentiment(self, ticker, days=7):
        """Get news sentiment for a ticker over the past few days."""
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
                    'top_headlines': []
                }
            
            # Calculate sentiment for each news item
            sentiments = []
            headlines = []
            
            for item in news:
                title = item.get('title', '')
                if title:
                    sentiment = self.sentiment_analyzer.polarity_scores(title)
                    sentiments.append(sentiment['compound'])
                    headlines.append({
                        'title': title,
                        'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d'),
                        'sentiment': sentiment['compound']
                    })
            
            # Calculate overall sentiment
            overall_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Determine sentiment trend
            if overall_sentiment > 0.2:
                sentiment_trend = 'very positive'
            elif overall_sentiment > 0:
                sentiment_trend = 'positive'
            elif overall_sentiment > -0.2:
                sentiment_trend = 'neutral'
            elif overall_sentiment > -0.5:
                sentiment_trend = 'negative'
            else:
                sentiment_trend = 'very negative'
            
            # Sort headlines by sentiment and get top 5
            headlines.sort(key=lambda x: abs(x['sentiment']), reverse=True)
            top_headlines = headlines[:5]
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_trend': sentiment_trend,
                'news_volume': len(news),
                'top_headlines': top_headlines
            }
        
        except Exception as e:
            logger.error(f"Error getting news sentiment for {ticker}: {e}")
            return {
                'overall_sentiment': 0,
                'sentiment_trend': 'neutral',
                'news_volume': 0,
                'top_headlines': []
            }
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for the data."""
        df = data.copy()
        
        # Moving Averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD_fast'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['MACD_slow'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['MACD_fast'] - df['MACD_slow']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA20']
        
        # Price momentum
        df['momentum_1d'] = df['Close'].pct_change(periods=1)
        df['momentum_5d'] = df['Close'].pct_change(periods=5)
        df['momentum_10d'] = df['Close'].pct_change(periods=10)
        df['momentum_20d'] = df['Close'].pct_change(periods=20)
        
        # Volatility
        df['volatility_20d'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean() * 100
        
        # Fill NaN values with 0 to avoid issues
        df = df.fillna(0)
        
        return df
    
    def detect_market_regime(self, data):
        """Detect the current market regime (trending, mean-reverting, volatile)."""
        df = data.copy()
        
        # Calculate returns
        returns = df['Close'].pct_change().dropna()
        
        # Calculate volatility
        volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
        
        # Calculate autocorrelation (for mean reversion)
        autocorr = returns.autocorr(lag=1)
        
        # Calculate trend strength
        sma20 = df['SMA20'].iloc[-1]
        sma50 = df['SMA50'].iloc[-1]
        sma200 = df['SMA200'].iloc[-1]
        
        price = df['Close'].iloc[-1]
        price_20d_ago = df['Close'].iloc[-21] if len(df) > 21 else df['Close'].iloc[0]
        price_50d_ago = df['Close'].iloc[-51] if len(df) > 51 else df['Close'].iloc[0]
        
        trend_strength = 0
        if price > sma20:
            trend_strength += 1
        if price > sma50:
            trend_strength += 1
        if price > sma200:
            trend_strength += 1
        if sma20 > sma50:
            trend_strength += 1
        if sma50 > sma200:
            trend_strength += 1
        if price > price_20d_ago:
            trend_strength += 1
        if price > price_50d_ago:
            trend_strength += 1
        
        # Determine regime
        if volatility > 30:
            regime = 'volatile'
        elif trend_strength >= 5:
            regime = 'trending'
        elif autocorr < -0.2:
            regime = 'mean-reverting'
        else:
            regime = 'neutral'
        
        return {
            'regime': regime,
            'trend_strength': trend_strength / 7,  # Normalize to 0-1
            'volatility': volatility,
            'autocorrelation': autocorr
        }
    
    def prepare_features(self, df):
        """Prepare features for the prediction model."""
        # Select relevant features
        features = df[['SMA20', 'SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_signal', 
                      '%K', '%D', 'ATR', 'Volume_ratio', 'momentum_1d', 'momentum_5d', 
                      'momentum_10d', 'momentum_20d', 'volatility_20d']].copy()
        
        # Add price relative to moving averages
        features['price_to_sma20'] = df['Close'] / df['SMA20'] - 1
        features['price_to_sma50'] = df['Close'] / df['SMA50'] - 1
        features['price_to_sma200'] = df['Close'] / df['SMA200'] - 1
        
        # Add Bollinger Band position
        features['bb_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Add day of week (to capture weekly patterns)
        features['day_of_week'] = pd.to_datetime(df.index).dayofweek
        
        # One-hot encode day of week
        for i in range(5):  # 0-4 for Monday-Friday
            features[f'day_{i}'] = (features['day_of_week'] == i).astype(int)
        
        # Drop original day of week column
        features.drop('day_of_week', axis=1, inplace=True)
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        return features
    
    def train_prediction_model(self, df, forecast_horizon=5):
        """Train a model to predict future price movements."""
        # Prepare features
        features = self.prepare_features(df)
        
        # Create target variable (future returns)
        target = df['Close'].pct_change(periods=forecast_horizon).shift(-forecast_horizon)
        
        # Remove NaN values
        valid_idx = ~target.isna()
        X = features[valid_idx].values
        y = target[valid_idx].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model
    
    def predict_future_movement(self, df, model, days=5):
        """Predict future price movement."""
        # Prepare features for the most recent data point
        features = self.prepare_features(df)
        latest_features = features.iloc[-1].values.reshape(1, -1)
        
        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        predicted_return = model.predict(latest_features_scaled)[0]
        
        # Calculate predicted price
        latest_price = df['Close'].iloc[-1]
        predicted_price = latest_price * (1 + predicted_return)
        
        # Calculate prediction confidence based on feature importance
        feature_importances = model.feature_importances_
        top_features_idx = np.argsort(feature_importances)[-5:]  # Top 5 features
        confidence = np.mean(feature_importances[top_features_idx]) * 10  # Scale to 0-1
        
        return {
            'predicted_return': predicted_return * 100,  # Convert to percentage
            'predicted_price': predicted_price,
            'confidence': min(confidence, 1.0),  # Cap at 1.0
            'forecast_horizon': days
        }
    
    def generate_entry_exit_points(self, df, prediction, regime_info):
        """Generate entry and exit points based on prediction and market regime."""
        latest_price = df['Close'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        
        # Adjust strategy based on market regime
        regime = regime_info['regime']
        
        if regime == 'trending':
            # In trending markets, use momentum-based entries and trailing stops
            if prediction['predicted_return'] > 2:  # Strong upward prediction
                entry_price = latest_price
                stop_loss = entry_price - 2 * latest_atr
                take_profit = entry_price + prediction['predicted_return'] / 100 * entry_price
                trailing_stop = True
                entry_condition = "Enter long position immediately as market is trending and prediction is positive"
                exit_condition = "Exit when price falls below trailing stop (initially set at 2 ATR below entry)"
            elif prediction['predicted_return'] < -2:  # Strong downward prediction
                entry_price = latest_price
                stop_loss = entry_price + 2 * latest_atr
                take_profit = entry_price + prediction['predicted_return'] / 100 * entry_price
                trailing_stop = True
                entry_condition = "Enter short position immediately as market is trending down"
                exit_condition = "Exit when price rises above trailing stop (initially set at 2 ATR above entry)"
            else:  # Weak prediction
                return None  # No trade in trending market with weak prediction
        
        elif regime == 'mean-reverting':
            # In mean-reverting markets, look for oversold/overbought conditions
            rsi = df['RSI'].iloc[-1]
            
            if prediction['predicted_return'] > 1 and rsi < 30:  # Oversold with positive prediction
                entry_price = latest_price
                stop_loss = entry_price - 1.5 * latest_atr
                take_profit = entry_price + 2 * latest_atr
                trailing_stop = False
                entry_condition = "Enter long position as market is oversold (RSI < 30) in mean-reverting regime"
                exit_condition = "Exit at take profit (2 ATR above entry) or stop loss (1.5 ATR below entry)"
            elif prediction['predicted_return'] < -1 and rsi > 70:  # Overbought with negative prediction
                entry_price = latest_price
                stop_loss = entry_price + 1.5 * latest_atr
                take_profit = entry_price - 2 * latest_atr
                trailing_stop = False
                entry_condition = "Enter short position as market is overbought (RSI > 70) in mean-reverting regime"
                exit_condition = "Exit at take profit (2 ATR below entry) or stop loss (1.5 ATR above entry)"
            else:
                return None  # No trade in mean-reverting market without oversold/overbought condition
        
        elif regime == 'volatile':
            # In volatile markets, use wider stops and take profits
            if abs(prediction['predicted_return']) > 3 and prediction['confidence'] > 0.6:
                # Only trade with high confidence in volatile markets
                if prediction['predicted_return'] > 0:
                    entry_price = latest_price
                    stop_loss = entry_price - 3 * latest_atr
                    take_profit = entry_price + 4 * latest_atr
                    trailing_stop = False
                    entry_condition = "Enter long position with high confidence despite volatile market"
                    exit_condition = "Exit at take profit (4 ATR above entry) or stop loss (3 ATR below entry)"
                else:
                    entry_price = latest_price
                    stop_loss = entry_price + 3 * latest_atr
                    take_profit = entry_price - 4 * latest_atr
                    trailing_stop = False
                    entry_condition = "Enter short position with high confidence despite volatile market"
                    exit_condition = "Exit at take profit (4 ATR below entry) or stop loss (3 ATR above entry)"
            else:
                return None  # Avoid trading in volatile markets without high confidence
        
        else:  # Neutral regime
            # In neutral markets, use balanced approach
            if prediction['predicted_return'] > 1.5:
                entry_price = latest_price
                stop_loss = entry_price - 1.5 * latest_atr
                take_profit = entry_price + 2 * latest_atr
                trailing_stop = False
                entry_condition = "Enter long position based on positive prediction in neutral market"
                exit_condition = "Exit at take profit (2 ATR above entry) or stop loss (1.5 ATR below entry)"
            elif prediction['predicted_return'] < -1.5:
                entry_price = latest_price
                stop_loss = entry_price + 1.5 * latest_atr
                take_profit = entry_price - 2 * latest_atr
                trailing_stop = False
                entry_condition = "Enter short position based on negative prediction in neutral market"
                exit_condition = "Exit at take profit (2 ATR below entry) or stop loss (1.5 ATR above entry)"
            else:
                return None  # No trade in neutral market with weak prediction
        
        # Calculate risk-reward ratio
        if prediction['predicted_return'] > 0:
            risk = (entry_price - stop_loss) / entry_price * 100
            reward = (take_profit - entry_price) / entry_price * 100
        else:
            risk = (stop_loss - entry_price) / entry_price * 100
            reward = (entry_price - take_profit) / entry_price * 100
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'direction': 'long' if prediction['predicted_return'] > 0 else 'short',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': trailing_stop,
            'risk_reward_ratio': risk_reward_ratio,
            'entry_condition': entry_condition,
            'exit_condition': exit_condition
        }
    
    def calculate_position_size(self, account_size, risk_per_trade, entry_price, stop_loss):
        """Calculate appropriate position size based on risk management."""
        risk_amount = account_size * (risk_per_trade / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = risk_amount / risk_per_share
        return position_size
    
    def generate_market_narrative(self, ticker, fundamentals, news_sentiment, regime_info, prediction):
        """Generate a qualitative market narrative explaining the context and outlook."""
        # Company description
        company_description = f"{ticker} is currently trading at a market cap of ${fundamentals.get('market_cap', 0) / 1e9:.2f}B "
        
        if fundamentals.get('pe_ratio', 0) > 0:
            company_description += f"with a P/E ratio of {fundamentals.get('pe_ratio', 0):.2f}. "
        else:
            company_description += "with negative earnings. "
        
        # Fundamental analysis
        fundamental_analysis = "From a fundamental perspective, "
        
        if fundamentals.get('revenue_growth', 0) > 10:
            fundamental_analysis += f"the company shows strong revenue growth of {fundamentals.get('revenue_growth', 0):.2f}% "
        elif fundamentals.get('revenue_growth', 0) > 0:
            fundamental_analysis += f"the company shows moderate revenue growth of {fundamentals.get('revenue_growth', 0):.2f}% "
        else:
            fundamental_analysis += "the company is experiencing declining revenues "
        
        if fundamentals.get('profit_margin', 0) > 15:
            fundamental_analysis += f"with a healthy profit margin of {fundamentals.get('profit_margin', 0):.2f}%. "
        elif fundamentals.get('profit_margin', 0) > 0:
            fundamental_analysis += f"with a modest profit margin of {fundamentals.get('profit_margin', 0):.2f}%. "
        else:
            fundamental_analysis += "but is currently unprofitable. "
        
        # News sentiment
        news_analysis = "Recent news sentiment is "
        
        if news_sentiment['sentiment_trend'] == 'very positive':
            news_analysis += f"very positive with {news_sentiment['news_volume']} recent news items showing enthusiasm about the company. "
        elif news_sentiment['sentiment_trend'] == 'positive':
            news_analysis += f"positive with {news_sentiment['news_volume']} recent news items showing optimism. "
        elif news_sentiment['sentiment_trend'] == 'neutral':
            news_analysis += f"neutral across {news_sentiment['news_volume']} recent news items. "
        elif news_sentiment['sentiment_trend'] == 'negative':
            news_analysis += f"negative with {news_sentiment['news_volume']} recent news items showing concerns. "
        else:
            news_analysis += f"very negative with {news_sentiment['news_volume']} recent news items showing significant worries about the company. "
        
        if news_sentiment['top_headlines']:
            top_headline = news_sentiment['top_headlines'][0]['title']
            news_analysis += f"A key headline is: \"{top_headline}\". "
        
        # Market regime
        regime_analysis = f"The market for {ticker} is currently in a {regime_info['regime']} regime "
        
        if regime_info['regime'] == 'trending':
            regime_analysis += f"with strong directional movement and a trend strength of {regime_info['trend_strength']*100:.2f}%. "
        elif regime_info['regime'] == 'mean-reverting':
            regime_analysis += "showing tendency to revert to the mean after price movements. "
        elif regime_info['regime'] == 'volatile':
            regime_analysis += f"with high volatility of {regime_info['volatility']:.2f}%. "
        else:
            regime_analysis += "with no clear directional bias. "
        
        # Future outlook
        future_outlook = "Looking forward, "
        
        if prediction['predicted_return'] > 3:
            future_outlook += f"the model predicts a strong positive movement of {prediction['predicted_return']:.2f}% "
        elif prediction['predicted_return'] > 0:
            future_outlook += f"the model predicts a modest positive movement of {prediction['predicted_return']:.2f}% "
        elif prediction['predicted_return'] > -3:
            future_outlook += f"the model predicts a modest negative movement of {abs(prediction['predicted_return']):.2f}% "
        else:
            future_outlook += f"the model predicts a strong negative movement of {abs(prediction['predicted_return']):.2f}% "
        
        future_outlook += f"over the next {prediction['forecast_horizon']} trading days with a confidence level of {prediction['confidence']*100:.2f}%. "
        
        # Combine all sections
        narrative = company_description + fundamental_analysis + news_analysis + regime_analysis + future_outlook
        
        return narrative
    
    def generate_forward_looking_strategy(self, ticker, account_size=100000, risk_per_trade=1):
        """Generate a complete forward-looking trading strategy with qualitative analysis."""
        try:
            # Get market data
            data = self.get_market_data(ticker)
            if data is None:
                return {
                    'success': False,
                    'error': f"Could not retrieve market data for {ticker}"
                }
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(data)
            
            # Get fundamental data
            fundamentals = self.get_company_fundamentals(ticker)
            
            # Get news sentiment
            news_sentiment = self.get_news_sentiment(ticker)
            
            # Detect market regime
            regime_info = self.detect_market_regime(df)
            
            # Train prediction model
            model = self.train_prediction_model(df)
            
            # Predict future movement
            prediction = self.predict_future_movement(df, model)
            
            # Generate entry/exit points
            trade_plan = self.generate_entry_exit_points(df, prediction, regime_info)
            
            # If no trade plan was generated, return early
            if trade_plan is None:
                return {
                    'success': True,
                    'ticker': ticker,
                    'recommendation': 'no_trade',
                    'reason': f"No favorable setup found in current {regime_info['regime']} market regime",
                    'prediction': prediction,
                    'regime': regime_info,
                    'fundamentals': fundamentals,
                    'news_sentiment': news_sentiment
                }
            
            # Calculate position size
            position_size = self.calculate_position_size(
                account_size, 
                risk_per_trade, 
                trade_plan['entry_price'], 
                trade_plan['stop_loss']
            )
            
            # Generate market narrative
            narrative = self.generate_market_narrative(
                ticker, 
                fundamentals, 
                news_sentiment, 
                regime_info, 
                prediction
            )
            
            # Determine overall recommendation
            if trade_plan['direction'] == 'long':
                recommendation = 'buy'
            else:
                recommendation = 'sell'
            
            # Compile complete strategy
            strategy = {
                'success': True,
                'ticker': ticker,
                'date_generated': datetime.now().strftime('%Y-%m-%d'),
                'recommendation': recommendation,
                'confidence': prediction['confidence'],
                'prediction': {
                    'direction': trade_plan['direction'],
                    'predicted_return': prediction['predicted_return'],
                    'predicted_price': prediction['predicted_price'],
                    'forecast_horizon': prediction['forecast_horizon'],
                    'confidence': prediction['confidence']
                },
                'trade_plan': {
                    'entry_price': trade_plan['entry_price'],
                    'stop_loss': trade_plan['stop_loss'],
                    'take_profit': trade_plan['take_profit'],
                    'trailing_stop': trade_plan['trailing_stop'],
                    'risk_reward_ratio': trade_plan['risk_reward_ratio'],
                    'position_size': position_size,
                    'entry_condition': trade_plan['entry_condition'],
                    'exit_condition': trade_plan['exit_condition']
                },
                'market_analysis': {
                    'regime': regime_info['regime'],
                    'trend_strength': regime_info['trend_strength'],
                    'volatility': regime_info['volatility'],
                    'current_price': df['Close'].iloc[-1],
                    'sma20': df['SMA20'].iloc[-1],
                    'sma50': df['SMA50'].iloc[-1],
                    'rsi': df['RSI'].iloc[-1],
                    'macd': df['MACD'].iloc[-1],
                    'macd_signal': df['MACD_signal'].iloc[-1]
                },
                'fundamental_analysis': fundamentals,
                'news_sentiment': {
                    'overall_sentiment': news_sentiment['overall_sentiment'],
                    'sentiment_trend': news_sentiment['sentiment_trend'],
                    'news_volume': news_sentiment['news_volume'],
                    'top_headlines': news_sentiment['top_headlines'][:3] if news_sentiment['top_headlines'] else []
                },
                'narrative': narrative
            }
            
            return strategy
        
        except Exception as e:
            logger.error(f"Error generating forward-looking strategy for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    strategy_generator = ForwardLookingStrategyGenerator()
    strategy = strategy_generator.generate_forward_looking_strategy("AAPL")
    
    if strategy['success']:
        print(f"Strategy for {strategy['ticker']}:")
        print(f"Recommendation: {strategy['recommendation'].upper()} with {strategy['confidence']*100:.2f}% confidence")
        print(f"Predicted return: {strategy['prediction']['predicted_return']:.2f}% over {strategy['prediction']['forecast_horizon']} days")
        print(f"Entry price: ${strategy['trade_plan']['entry_price']:.2f}")
        print(f"Stop loss: ${strategy['trade_plan']['stop_loss']:.2f}")
        print(f"Take profit: ${strategy['trade_plan']['take_profit']:.2f}")
        print(f"Risk-reward ratio: {strategy['trade_plan']['risk_reward_ratio']:.2f}")
        print(f"Position size: {strategy['trade_plan']['position_size']:.0f} shares")
        print("\nMarket Narrative:")
        print(strategy['narrative'])
    else:
        print(f"Error: {strategy['error']}")
