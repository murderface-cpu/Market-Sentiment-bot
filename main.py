import os
import logging
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
import pandas as pd
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton, WebAppInfo, InputFile
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from textblob import TextBlob
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import nest_asyncio
import asyncio
nest_asyncio.apply()

import tweepy
import praw
import requests
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from datetime import timezone
import time
from typing import Dict, List, Optional

from aiohttp import web

# Setup logging
LOG_FILE = "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ['TELEGRAM_BOT_TOKEN'] = '6275443131:AAEH83H5OKBv9a_rfdEP1sjlmiFTAR8AxCc'
os.environ['TWITTER_BEARER_TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAACvArAEAAAAAlBQfl0tEgogqxywzWdYSTasVi5s%3DA5kNHi2x3QtSqvustU98wi4AU7e17kqREz5Io7VbB37c5WTWxr'
# Reddit API (free)
os.environ['REDDIT_CLIENT_ID'] ='bLMni0816wQQLvDWIWk9QQ'
os.environ['REDDIT_CLIENT_SECRET'] ='QQe7bEDGja89k3POkbQJLUfup5nbcg'
os.environ['REDDIT_USER_AGENT'] = 'MarketSentimentBot/1.0'
# News API (free tier: 1000 requests/month)
os.environ['NEWS_API_KEY'] = '7c397cd6b8b74e5db62726948308a3d7'

class MarketSentimentBot:
    def __init__(self, token: str):
        self.token = token
        self.db_path = "sentiment_bot.db"
        self.premium_users = set()
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # API configurations (add these to your environment variables)
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'MarketSentimentBot/1.0')
        self.news_api_key = os.getenv('NEWS_API_KEY')

        # Initialize APIs
        self.init_apis()
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for user data and analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                subscription_type TEXT DEFAULT 'free',
                subscription_end DATE,
                queries_today INTEGER DEFAULT 0,
                last_query_date DATE,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT,
                sentiment_score REAL,
                confidence REAL,
                sources_analyzed INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        conn.commit()
        conn.close()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        user = update.effective_user
        self.register_user(user.id, user.username)

        welcome_message = f"""
🔮 **Market Sentiment Oracle** 🔮

Welcome {user.first_name}! I analyze real-time social media buzz, news sentiment, and market indicators to give you edge in trading decisions.

📊 **What I can do:**
• Analyze sentiment for any stock/crypto
• Generate beautiful sentiment reports
• Track sentiment trends over time
• Send daily market mood updates
• Provide trading signals based on sentiment

🆓 **Free Plan:** 5 analyses per day
💎 **Premium Plan:** Unlimited + alerts + API access

Choose an option below to get started!
        """

        keyboard = [
            [InlineKeyboardButton("📈 Analyze Stock/Crypto", callback_data="analyze_stock")],
            [InlineKeyboardButton("Check Application Health", web_app=WebAppInfo(url="https://market-sentiment-bot.onrender.com/health"))],
            [InlineKeyboardButton("🔔 Setup Alerts", callback_data="setup_alerts")],
            [InlineKeyboardButton("💎 Upgrade to Premium", callback_data="upgrade_premium")],
            [InlineKeyboardButton("📧 Email Settings", callback_data="email_settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode='Markdown')

    def register_user(self, user_id: int, username: str):
        """Register new user in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR IGNORE INTO users (user_id, username)
            VALUES (?, ?)
        ''', (user_id, username))

        conn.commit()
        conn.close()

    async def analyze_sentiment(self, symbol: str, user_id: int) -> Dict:
        """Analyze sentiment for a given stock/crypto symbol"""
        try:
            # Simulate sentiment analysis (in real implementation, use Twitter API, Reddit API, news APIs)
            sentiment_sources = await self.gather_sentiment_data(symbol)

            # Calculate overall sentiment
            sentiment_score = self.calculate_sentiment_score(sentiment_sources)
            confidence = self.calculate_confidence(sentiment_sources)
            # Get price data
            price_data = await self.get_price_data(symbol)

            # Generate visualization
            chart_url = await self.generate_sentiment_chart_v2(symbol, sentiment_score, price_data)
            logger.error(chart_url)
            #await self.generate_and_send_chart(symbol, sentiment_score, price_data)

            # Store in database
            self.store_sentiment_data(user_id, symbol, sentiment_score, confidence, len(sentiment_sources))

            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'sources_analyzed': len(sentiment_sources),
                'price_data': price_data,
                'chart_url': chart_url,
                'recommendation': self.get_trading_recommendation(sentiment_score, confidence)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return None

    def init_apis(self):
        """Initialize API clients"""
        try:
            # Twitter API v2 client
            if self.twitter_bearer_token:
                self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)

            # Reddit API client
            if all([self.reddit_client_id, self.reddit_client_secret]):
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )

            # News API client
            if self.news_api_key:
                self.news_api = NewsApiClient(api_key=self.news_api_key)

        except Exception as e:
            logger.error(f"Error initializing APIs: {e}")

    async def gather_sentiment_data(self, symbol: str) -> List[Dict]:
        """Gather real sentiment data from various sources"""
        all_sources = []

        try:
            # Get Twitter sentiment
            twitter_data = await self.get_twitter_sentiment(symbol)
            all_sources.extend(twitter_data)

            # Get Reddit sentiment
            reddit_data = await self.get_reddit_sentiment(symbol)
            all_sources.extend(reddit_data)

            # Get news sentiment
            news_data = await self.get_news_sentiment(symbol)
            all_sources.extend(news_data)

            # Get StockTwits sentiment (if available)
            stocktwits_data = await self.get_stocktwits_sentiment(symbol)
            all_sources.extend(stocktwits_data)

        except Exception as e:
            logger.error(f"Error gathering sentiment data: {e}")

        return all_sources

    async def get_twitter_sentiment(self, symbol: str) -> List[Dict]:
        """Get sentiment from Twitter using API v2"""
        sources = []

        if not hasattr(self, 'twitter_client'):
            return sources

        try:
            # Search queries for the symbol
            queries = [
                f"${symbol} -is:retweet lang:en",
                f"{symbol} stock -is:retweet lang:en",
                f"{symbol} buy sell hold -is:retweet lang:en"
            ]

            for query in queries:
                tweets = tweepy.Paginator(
                    self.twitter_client.search_recent_tweets,
                    query=query,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics', 'context_annotations']
                ).flatten(limit=200)

                for tweet in tweets:
                    if tweet.text:
                        # Clean tweet text
                        clean_text = self.clean_text(tweet.text)

                        # Get sentiment score
                        sentiment_score = self.get_sentiment_score(clean_text)

                        sources.append({
                            'text': clean_text,
                            'sentiment': sentiment_score,
                            'source': 'twitter',
                            'timestamp': tweet.created_at,
                            'engagement': tweet.public_metrics.get('like_count', 0) +
                                        tweet.public_metrics.get('retweet_count', 0)
                        })

        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")

        return sources

    async def get_reddit_sentiment(self, symbol: str) -> List[Dict]:
        """Get sentiment from Reddit"""
        sources = []

        if not hasattr(self, 'reddit'):
            return sources

        try:
            # Subreddits to search
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis', 'StockMarket']

            # If it's a crypto symbol, add crypto subreddits
            if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'DOGE']):
                subreddits.extend(['cryptocurrency', 'CryptoMarkets', 'Bitcoin', 'ethereum'])

            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Search for posts containing the symbol
                    for post in subreddit.search(symbol, limit=300, time_filter='day'):
                        # Process post title and content
                        text = f"{post.title} {post.selftext}"
                        clean_text = self.clean_text(text)

                        if len(clean_text) > 10:  # Filter out very short posts
                            sentiment_score = self.get_sentiment_score(clean_text)

                            sources.append({
                                'text': clean_text[:500],  # Limit text length
                                'sentiment': sentiment_score,
                                'source': f'reddit_{subreddit_name}',
                                'timestamp': datetime.fromtimestamp(post.created_utc),
                                'engagement': post.score + post.num_comments
                            })

                        # Also process top comments
                        post.comments.replace_more(limit=0)
                        for comment in post.comments[:50]:  # Top 50 comments
                            if hasattr(comment, 'body') and len(comment.body) > 20:
                                clean_comment = self.clean_text(comment.body)
                                comment_sentiment = self.get_sentiment_score(clean_comment)

                                sources.append({
                                    'text': clean_comment[:300],
                                    'sentiment': comment_sentiment,
                                    'source': f'reddit_{subreddit_name}_comment',
                                    'timestamp': datetime.fromtimestamp(comment.created_utc),
                                    'engagement': comment.score
                                })

                except Exception as e:
                    logger.error(f"Error processing subreddit {subreddit_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")

        return sources

    async def get_news_sentiment(self, symbol: str) -> List[Dict]:
        """Get sentiment from news articles"""
        sources = []

        if not hasattr(self, 'news_api'):
            return sources

        try:
            # Get company name for better search results
            company_name = await self.get_company_name(symbol)

            # Search queries
            queries = [symbol, company_name] if company_name else [symbol]

            for query in queries:
                if not query:
                    continue

                # Get news from last 24 hours
                articles = self.news_api.get_everything(
                    q=query,
                    language='en',
                    sort_by='publishedAt',
                    from_param=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    page_size=100
                )

                for article in articles.get('articles', []):
                    # Combine title and description
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    clean_text = self.clean_text(text)

                    if len(clean_text) > 20:
                        sentiment_score = self.get_sentiment_score(clean_text)

                        sources.append({
                            'text': clean_text,
                            'sentiment': sentiment_score,
                            'source': 'news',
                            'timestamp': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                            'engagement': 1,  # All news articles get base engagement
                            'source_name': article.get('source', {}).get('name', 'Unknown')
                        })

        except Exception as e:
            logger.error(f"Error fetching news data: {e}")

        return sources

    async def get_stocktwits_sentiment(self, symbol: str) -> List[Dict]:
        """Get sentiment from StockTwits (free API)"""
        sources = []

        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        for message in data.get('messages', []):
                            clean_text = self.clean_text(message.get('body', ''))

                            if len(clean_text) > 10:
                                sentiment_score = self.get_sentiment_score(clean_text)

                                sources.append({
                                    'text': clean_text,
                                    'sentiment': sentiment_score,
                                    'source': 'stocktwits',
                                    'timestamp': datetime.fromisoformat(message['created_at'].replace('Z', '+00:00')),
                                    'engagement': message.get('likes', {}).get('total', 0)
                                })

        except Exception as e:
            logger.error(f"Error fetching StockTwits data: {e}")

        return sources

    def clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags for cleaner analysis (keep the content)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def get_sentiment_score(self, text: str) -> float:
        """Get sentiment score using VADER"""
        if not text:
            return 0.0

        try:
            # Use VADER sentiment analyzer
            scores = self.sentiment_analyzer.polarity_scores(text)

            # Return compound score (normalized between -1 and 1)
            return scores['compound']

        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return 0.0

    async def get_company_name(self, symbol: str) -> Optional[str]:
        """Get company name from symbol for better news search"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('longName') or info.get('shortName')
        except:
            return None

    def calculate_sentiment_score(self, sources: List[Dict]) -> float:
        """Calculate weighted sentiment score based on engagement"""
        if not sources:
            return 0.0

        total_weighted_sentiment = 0
        total_weight = 0

        for source in sources:
            # Weight by engagement (likes, retweets, upvotes, etc.)
            engagement = source.get('engagement', 1)
            weight = min(max(1, engagement), 100)  # Cap weight between 1-100

            # Also weight by recency (more recent = higher weight)
            hours_old = (datetime.now(timezone.utc) - source['timestamp'].replace(tzinfo=timezone.utc)).total_seconds() / 3600
            recency_weight = max(0.1, 1 - (hours_old / 24))  # Decay over 24 hours

            final_weight = weight * recency_weight

            total_weighted_sentiment += source['sentiment'] * final_weight
            total_weight += final_weight

        return total_weighted_sentiment / total_weight if total_weight > 0 else 0.0

    def calculate_confidence(self, sources: List[Dict]) -> float:
        """Calculate confidence based on source volume, agreement, and quality"""
        if not sources:
            return 0.0

        # Volume factor (more sources = higher confidence)
        volume_factor = min(len(sources) / 200, 1.0)  # Cap at 200 sources

        # Agreement factor (less variance = higher confidence)
        sentiments = [s['sentiment'] for s in sources]
        variance = pd.Series(sentiments).var()
        agreement_factor = max(0, 1 - min(variance * 2, 1))  # Scale variance

        # Source diversity factor (more diverse sources = higher confidence)
        unique_sources = len(set(s['source'] for s in sources))
        diversity_factor = min(unique_sources / 5, 1.0)  # Cap at 5 different source types

        # Quality factor (higher engagement = higher confidence)
        avg_engagement = sum(s.get('engagement', 1) for s in sources) / len(sources)
        quality_factor = min(avg_engagement / 50, 1.0)  # Cap at 50 avg engagement

        # Combine factors
        confidence = (volume_factor * 0.4 + agreement_factor * 0.3 +
                     diversity_factor * 0.2 + quality_factor * 0.1)

        return max(0.1, min(0.95, confidence))

    async def get_price_data(self, symbol: str) -> Dict:
        """Enhanced price data with technical indicators"""
        try:
            ticker = yf.Ticker(symbol)

            # Get basic info
            info = ticker.info

            # Get historical data (extended period for technical analysis)
            hist_1d = ticker.history(period="1d", interval="5m")  # Intraday
            hist_5d = ticker.history(period="5d", interval="1h")  # Short term
            hist_1mo = ticker.history(period="1mo", interval="1d")  # Medium term

            # Calculate technical indicators
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            # Volume analysis
            avg_volume = hist_1mo['Volume'].mean() if not hist_1mo.empty else 0
            current_volume = info.get('volume', 0)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Price momentum
            price_change_24h = info.get('regularMarketChangePercent', 0)

            # Support/Resistance levels (simplified)
            if not hist_1mo.empty:
                resistance = hist_1mo['High'].quantile(0.95)
                support = hist_1mo['Low'].quantile(0.05)
            else:
                resistance = support = current_price

            return {
                'current_price': current_price,
                'change_percent': price_change_24h,
                'volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'resistance': resistance,
                'support': support,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'history_1d': hist_1d.to_dict('records') if not hist_1d.empty else [],
                'history_5d': hist_5d.to_dict('records') if not hist_5d.empty else [],
                'history_1mo': hist_1mo.to_dict('records') if not hist_1mo.empty else []
            }

        except Exception as e:
            logger.error(f"Error fetching enhanced price data for {symbol}: {e}")
            return {'current_price': 0, 'change_percent': 0, 'volume': 0}

    # Additional method to add for better trading recommendations
    def get_trading_recommendation(self, sentiment: float, confidence: float, price_data: Dict = None) -> str:
        """Enhanced trading recommendation with price analysis"""
        if confidence < 0.3:
            return "⚠️ **LOW CONFIDENCE** - Wait for clearer signals"

        # Base recommendation on sentiment
        if sentiment > 0.4:
            base_rec = "🟢 **STRONG BULLISH**"
            action = "Consider strong buying opportunities"
        elif sentiment > 0.2:
            base_rec = "🟢 **BULLISH**"
            action = "Consider buying opportunities"
        elif sentiment > -0.1:
            base_rec = "🟡 **NEUTRAL**"
            action = "Hold current positions"
        elif sentiment > -0.3:
            base_rec = "🔴 **BEARISH**"
            action = "Consider selling or reducing positions"
        else:
            base_rec = "🔴 **STRONG BEARISH**"
            action = "Consider selling or shorting opportunities"

        # Add price-based modifiers if available
        modifiers = []
        if price_data:
            volume_ratio = price_data.get('volume_ratio', 1)
            if volume_ratio > 2:
                modifiers.append("High volume confirms trend")
            elif volume_ratio < 0.5:
                modifiers.append("Low volume - trend may be weak")

            current_price = price_data.get('current_price', 0)
            resistance = price_data.get('resistance', 0)
            support = price_data.get('support', 0)

            if resistance and current_price > resistance * 0.95:
                modifiers.append("Near resistance level")
            elif support and current_price < support * 1.05:
                modifiers.append("Near support level")

        result = f"{base_rec} - {action}"
        if modifiers:
            result += f"\n📊 *{' • '.join(modifiers)}*"

        return result


    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    async def generate_and_send_chart(self, symbol: str, sentiment: float, price_data: Dict, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate chart and send as photo/document"""

        # Generate the chart
        fig = go.Figure()

        sentiment_color = '#4CAF50' if sentiment > 0 else '#F44336' if sentiment < -0.2 else '#FF9800'

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=sentiment * 100,
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [-100, -20], 'color': '#FFCDD2'},
                    {'range': [-20, 20], 'color': '#FFF3E0'},
                    {'range': [20, 100], 'color': '#C8E6C9'}
                ]
            }
        ))

        fig.update_layout(
            height=400,
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            title=f"Sentiment Analysis: {symbol}"
        )

        # Save as image instead of HTML
        chart_filename = f"sentiment_{symbol}_{int(datetime.now().timestamp())}.png"
        fig.write_image(chart_filename, width=800, height=400)

        # Send as photo
        with open(chart_filename, 'rb') as chart_file:
            await update.message.reply_photo(
                photo=InputFile(chart_file),
                caption=f"📊 Sentiment Analysis for {symbol}\n💯 Score: {sentiment*100:.1f}%"
            )

        # Clean up file
        os.remove(chart_filename)

    # Fix: Updated sentiment chart generation
    async def generate_sentiment_chart(self, symbol: str, sentiment: float, price_data: Dict) -> str:
        """Generate beautiful sentiment visualization"""

        # Method 1: Create separate figures (Recommended)
        # Create sentiment gauge as standalone figure
        sentiment_color = '#4CAF50' if sentiment > 0 else '#F44336' if sentiment < -0.2 else '#FF9800'

        fig = go.Figure()

        # Add sentiment gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment * 100,
            domain={'x': [0, 1], 'y': [0.5, 1]},  # Top half
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [-100, -20], 'color': '#FFCDD2'},
                    {'range': [-20, 20], 'color': '#FFF3E0'},
                    {'range': [20, 100], 'color': '#C8E6C9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))

        # Add price chart in bottom half
        if price_data.get('history'):
            df = pd.DataFrame(price_data['history'])

            # Create x-axis for dates if available
            if 'Date' in df.columns:
                x_data = pd.to_datetime(df['Date'])
            else:
                x_data = df.index

            fig.add_trace(go.Scatter(
                x=x_data,
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#2196F3', width=2),
                yaxis='y2',
                xaxis='x2'
            ))

            # Configure layout for both plots
            fig.update_layout(
                xaxis2=dict(domain=[0, 1], anchor='y2', position=0),
                yaxis2=dict(domain=[0, 0.4], anchor='x2', title='Price ($)'),
            )

        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            title=dict(
                text=f"Market Sentiment Analysis: {symbol}",
                font=dict(size=20, color='white'),
                x=0.5
            )
        )

        # Save chart
        chart_filename = f"sentiment_{symbol}_{int(datetime.now().timestamp())}.html"
        fig.write_html(chart_filename)
        return chart_filename

    # Alternative Method 2: Use proper subplot configuration for indicators
    async def generate_sentiment_chart_v2(self, symbol: str, sentiment: float, price_data: Dict) -> str:
        """Alternative approach with proper subplot types"""

        # Create subplots with different types
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{symbol} Sentiment Analysis", "Price Movement"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4],
            specs=[[{"type": "indicator"}],
                  [{"type": "xy"}]]  # Specify subplot types
        )

        # Sentiment gauge
        sentiment_color = '#4CAF50' if sentiment > 0 else '#F44336' if sentiment < -0.2 else '#FF9800'
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment * 100,
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [-100, -20], 'color': '#FFCDD2'},
                    {'range': [-20, 20], 'color': '#FFF3E0'},
                    {'range': [20, 100], 'color': '#C8E6C9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ), row=1, col=1)

        # Price chart
        if price_data.get('history'):
            df = pd.DataFrame(price_data['history'])

            # Handle different date formats
            if 'Date' in df.columns:
                x_data = pd.to_datetime(df['Date'])
            else:
                x_data = df.index

            fig.add_trace(go.Scatter(
                x=x_data,
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#2196F3', width=2)
            ), row=2, col=1)

        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            title=dict(
                text=f"Market Sentiment Analysis: {symbol}",
                font=dict(size=20, color='white')
            )
        )

        # Save as PNG file
        chart_filename = f"sentiment_{symbol}_{int(datetime.now().timestamp())}.png"
        fig.write_image(
            chart_filename,
            format='png',
            width=800,
            height=600,
            scale=2  # Higher scale for better quality
        )

        return chart_filename

    # Method 3: Simple bar chart alternative (if gauge doesn't work)
    async def generate_sentiment_chart_simple(self, symbol: str, sentiment: float, price_data: Dict) -> str:
        """Simple bar chart version as fallback"""

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{symbol} Sentiment Score", "Price Movement"),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6]
        )

        # Sentiment bar chart
        sentiment_color = '#4CAF50' if sentiment > 0 else '#F44336' if sentiment < -0.2 else '#FF9800'
        fig.add_trace(go.Bar(
            x=['Sentiment'],
            y=[sentiment * 100],
            marker_color=sentiment_color,
            name='Sentiment Score',
            text=[f'{sentiment * 100:.1f}%'],
            textposition='auto',
        ), row=1, col=1)

        # Price chart
        if price_data.get('history'):
            df = pd.DataFrame(price_data['history'])
            fig.add_trace(go.Scatter(
                x=df.index if 'Date' not in df.columns else pd.to_datetime(df['Date']),
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#2196F3', width=2)
            ), row=2, col=1)

        # Update y-axis for sentiment to show range
        fig.update_yaxes(range=[-100, 100], row=1, col=1)
        fig.update_yaxes(title_text="Sentiment %", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)

        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            title=dict(
                text=f"Market Sentiment Analysis: {symbol}",
                font=dict(size=20, color='white')
            )
        )

        chart_filename = f"sentiment_{symbol}_{int(datetime.now().timestamp())}.html"
        fig.write_html(chart_filename)
        return chart_filename

    def store_sentiment_data(self, user_id: int, symbol: str, sentiment: float, confidence: float, sources: int):
        """Store sentiment analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO sentiment_history
            (user_id, symbol, sentiment_score, confidence, sources_analyzed)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, symbol, sentiment, confidence, sources))

        conn.commit()
        conn.close()

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query

        try:
            await query.answer()
            user_id = update.effective_user.id

            if query.data == "analyze_stock":
                await query.edit_message_text(
                    "📊 **Stock/Crypto Analysis**\n\n"
                    "Send me a ticker symbol (e.g., AAPL, TSLA, BTC-USD, ETH-USD) and I'll analyze the current market sentiment!\n\n"
                    "💡 *Tip: Use official ticker symbols for best results*",
                    parse_mode='Markdown'
                )
                context.user_data['expecting'] = 'symbol'

            elif query.data == "setup_alerts":
                keyboard = [
                    [InlineKeyboardButton("🔔 Sentiment Alerts", callback_data="sentiment_alerts")],
                    [InlineKeyboardButton("📈 Price Alerts", callback_data="price_alerts")],
                    [InlineKeyboardButton("📰 News Alerts", callback_data="news_alerts")],
                    [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back_to_menu")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    "🔔 **Alert Settings**\n\n"
                    "Set up custom alerts for your favorite stocks and cryptos:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )

            elif query.data == "upgrade_premium":
                await self.show_premium_options(query)

            elif query.data == "email_settings":
                await query.edit_message_text(
                    "📧 **Email Settings**\n\n"
                    "Send me your email address to receive:\n"
                    "• Daily market sentiment reports\n"
                    "• Weekly trend analysis\n"
                    "• Custom alert notifications\n\n"
                    "Type your email address:",
                    parse_mode='Markdown'
                )
                context.user_data['expecting'] = 'email'

            elif query.data == "back_to_menu":
                await self.show_main_menu(query)

            elif query.data.startswith("sub_"):
                await self.handle_subscription(query, context)

            elif query.data == "trial_premium":
                await self.activate_trial(query, context, user_id)

            elif query.data.startswith("alert_"):
                symbol = query.data.replace("alert_", "")
                await self.setup_alert(query, context, symbol)

            elif query.data.startswith("email_"):
                symbol = query.data.replace("email_", "")
                await self.send_email_report(query, context, symbol, user_id)

        except Exception as e:
            logger.error(f"Error in callback query handler: {e}")
            await query.edit_message_text(
                "❌ **Error**\n\nSomething went wrong. Please try again or use /start.",
                parse_mode='Markdown'
            )


    async def show_premium_options(self, query):
        """Show premium subscription options"""
        keyboard = [
            [InlineKeyboardButton("💎 Monthly - $9.99", callback_data="sub_monthly")],
            [InlineKeyboardButton("💎 Yearly - $99.99 (Save 17%)", callback_data="sub_yearly")],
            [InlineKeyboardButton("🆓 Try Free Premium (7 days)", callback_data="trial_premium")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_to_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        premium_text = """💎 **Premium Features**

🚀 **Unlimited Analysis** - No daily limits
📊 **Advanced Charts** - Technical indicators + sentiment
🔔 **Real-time Alerts** - Custom sentiment thresholds
📈 **Portfolio Tracking** - Monitor multiple positions
🤖 **API Access** - Integrate with your apps
📧 **Email Reports** - Daily market insights
📱 **Priority Support** - Fast response times

Choose your plan:"""

        try:
            await query.edit_message_text(
                premium_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error showing premium options: {e}")

    async def show_main_menu(self, query):
        """Show the main menu"""
        welcome_message = f"""
🔮 **Market Sentiment Oracle** 🔮

Welcome! I analyze real-time social media buzz, news sentiment, and market indicators to give you edge in trading decisions.

📊 **What I can do:**
• Analyze sentiment for any stock/crypto
• Generate beautiful sentiment reports
• Track sentiment trends over time
• Send daily market mood updates
• Provide trading signals based on sentiment

🆓 **Free Plan:** 5 analyses per day
💎 **Premium Plan:** Unlimited + alerts + API access

Choose an option below to get started!
        """
        keyboard = [
            [InlineKeyboardButton("📊 Analyze Stock/Crypto", callback_data="analyze_stock")],
            [InlineKeyboardButton("🔔 Setup Alerts", callback_data="setup_alerts")],
            [InlineKeyboardButton("💎 Upgrade Premium", callback_data="upgrade_premium")],
            [InlineKeyboardButton("📧 Email Settings", callback_data="email_settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await query.edit_message_text(
                welcome_message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error showing main menu: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        if not update.message or not update.message.text:
            return

        user_id = update.effective_user.id
        text = update.message.text.strip()
        expecting = context.user_data.get('expecting')

        try:
            if expecting == 'symbol':
                # Clean and validate symbol
                symbol = self.clean_symbol(text)
                if self.validate_symbol(symbol):
                    await self.process_symbol_analysis(update, context, symbol)
                else:
                    await update.message.reply_text(
                        "❌ **Invalid Symbol**\n\n"
                        "Please enter a valid ticker symbol (e.g., AAPL, TSLA, BTC-USD, ETH-USD)",
                        parse_mode='Markdown'
                    )

            elif expecting == 'email':
                email = text.strip().lower()
                if self.validate_email(email):
                    await self.process_email_setup(update, context, email)
                else:
                    await update.message.reply_text(
                        "❌ **Invalid Email**\n\n"
                        "Please enter a valid email address.",
                        parse_mode='Markdown'
                    )
            else:
                # Check if message looks like a symbol
                cleaned_text = self.clean_symbol(text)
                if len(cleaned_text) <= 6 and cleaned_text.isalpha():
                    context.user_data['expecting'] = 'symbol'
                    await self.process_symbol_analysis(update, context, cleaned_text)
                else:
                    # Default response
                    await update.message.reply_text(
                        "Use /start to see available options, or click 📈 Analyze to get started!"
                    )

        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            await update.message.reply_text(
                "❌ **Error**\n\nSomething went wrong. Please try again or use /start.",
                parse_mode='Markdown'
            )


    def clean_symbol(self, text: str) -> str:
        """Clean and format symbol input"""
        # Remove special characters except hyphens and periods
        symbol = re.sub(r'[^\w\-\.]', '', text.upper())
        # Handle common crypto formats
        if symbol.endswith('USD') and '-' not in symbol:
            symbol = symbol[:-3] + '-USD'
        return symbol

    def validate_symbol(self, symbol: str) -> bool:
        """Validate ticker symbol format"""
        if not symbol or len(symbol) < 1 or len(symbol) > 12:
            return False
        # Allow letters, numbers, hyphens, and periods
        return bool(re.match(r'^[A-Z0-9\-\.]+$', symbol))

    def validate_email(self, email: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    async def process_symbol_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Process stock symbol analysis request"""
        user_id = update.effective_user.id

        # Check user limits
        if not self.can_analyze(user_id):
            keyboard = [[InlineKeyboardButton("💎 Upgrade to Premium", callback_data="upgrade_premium")]
                       [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back_to_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "⚠️ **Daily Limit Reached**\n\n"
                "Free users get 5 analyses per day.\n"
                "Upgrade to Premium for unlimited access!",
                reply_markup=reply_markup
            )
            #return

        # Send "analyzing" message
        analyzing_msg = await update.message.reply_text(
            f"🔍 **Analyzing {symbol}...**\n\n"
            "📊 Gathering social media data...\n"
            "📈 Fetching price information...\n"
            "🧠 Computing sentiment score...\n\n"
            "*This may take 30-60 seconds*"
        )

        # Perform analysis
        analysis = await self.analyze_sentiment(symbol, user_id)

        if analysis:
            # Create detailed result message
            result_message = self.format_analysis_result(analysis)

            # Send photo directly
            with open(analysis['chart_url'], 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"📊 Sentiment Analysis for {symbol}"
                )
            # Create action buttons
            keyboard = [
                [
                    # InlineKeyboardButton("📊 View Chart", url=analysis['chart_url']),
                    [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back_to_menu")],
                    InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert_{symbol}")
                ],
                [
                    InlineKeyboardButton("📧 Email Report", callback_data=f"email_{symbol}"),
                    InlineKeyboardButton("📈 Analyze Another", callback_data="analyze_stock")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await analyzing_msg.edit_text(result_message, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await analyzing_msg.edit_text(
                f"❌ **Analysis Failed**\n\n"
                f"Could not analyze {symbol}. Please check the ticker symbol and try again."
            )

        context.user_data['expecting'] = None

    def format_analysis_result(self, analysis: Dict) -> str:
        """Format analysis results into a beautiful message"""
        symbol = analysis['symbol']
        sentiment = analysis['sentiment_score']
        confidence = analysis['confidence']
        sources = analysis['sources_analyzed']
        recommendation = analysis['recommendation']

        # Sentiment emoji and description
        if sentiment > 0.3:
            sentiment_emoji = "🟢"
            sentiment_desc = "Very Bullish"
        elif sentiment > 0.1:
            sentiment_emoji = "🟢"
            sentiment_desc = "Bullish"
        elif sentiment > -0.1:
            sentiment_emoji = "🟡"
            sentiment_desc = "Neutral"
        elif sentiment > -0.3:
            sentiment_emoji = "🔴"
            sentiment_desc = "Bearish"
        else:
            sentiment_emoji = "🔴"
            sentiment_desc = "Very Bearish"

        confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))

        result = f"""
🔮 **Market Sentiment Analysis**

**{symbol}** {sentiment_emoji}

**Sentiment Score:** {sentiment:.2f} ({sentiment_desc})
**Confidence:** {confidence:.1%} {confidence_bar}
**Sources Analyzed:** {sources:,} posts/articles

{recommendation}

📊 **Key Insights:**
• Social media buzz is {'high' if sources > 100 else 'moderate' if sources > 50 else 'low'}
• Sentiment trend: {'Improving' if sentiment > 0 else 'Declining' if sentiment < -0.2 else 'Stable'}
• Community confidence: {'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'}

⏰ *Analysis generated at {datetime.now().strftime('%H:%M UTC')}*
        """

        return result

    def can_analyze(self, user_id: int) -> bool:
        """Check if user can perform analysis (rate limiting)"""
        if user_id in self.premium_users:
            return True

        # Check daily limit for free users
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().date()
        cursor.execute('''
            SELECT queries_today, last_query_date FROM users WHERE user_id = ?
        ''', (user_id,))

        result = cursor.fetchone()
        if result:
            queries_today, last_query_date = result

            if str(last_query_date) != str(today):
                # New day, reset counter
                cursor.execute('''
                    UPDATE users SET queries_today = 0, last_query_date = ? WHERE user_id = ?
                ''', (today, user_id))
                queries_today = 0

            if queries_today >= 5:  # Free limit
                conn.close()
                return False

            # Increment counter
            cursor.execute('''
                UPDATE users SET queries_today = queries_today + 1 WHERE user_id = ?
            ''', (user_id,))

        conn.commit()
        conn.close()
        return True

    async def process_email_setup(self, update: Update, context: ContextTypes.DEFAULT_TYPE, email: str):
        """Process email setup"""
        user_id = update.effective_user.id

        # Basic email validation
        if '@' not in email or '.' not in email.split('@')[-1]:
            await update.message.reply_text(
                "❌ **Invalid Email**\n\n"
                "Please enter a valid email address."
            )
            return

        # Store email in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET email = ? WHERE user_id = ?
        ''', (email.lower(), user_id))
        conn.commit()
        conn.close()

        keyboard = [
            [InlineKeyboardButton("✅ Generate Sample Report", callback_data="sample_email")],
            [InlineKeyboardButton("📋 Copy Report to Clipboard", callback_data="copy_report")],
            [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back_to_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"✅ **Email Configured Successfully!**\n\n"
            f"📧 **Your email:** {email}\n\n"
            "You'll now receive:\n"
            "• Daily market sentiment summaries\n"
            "• Weekly trend reports\n"
            "• Custom alert notifications\n\n"
            "Generate a sample report below:",
            reply_markup=reply_markup
        )

        context.user_data['expecting'] = None
        
        
# Simple aiohttp server for health check and logs
async def handle_health(request):
    return web.Response(text="✅ Market Sentiment Bot is healthy!", status=200)

async def handle_logs(request):
    try:
        with open(LOG_FILE, "r") as f:
            content = f.read()[-5000:]  # Last 5000 characters
        return web.Response(text=content, content_type='text/plain')
    except FileNotFoundError:
        return web.Response(text="Log file not found", status=404)

async def start_http_server():
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 8080))
    
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/logs", handle_logs)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Bind to 0.0.0.0 to accept connections from anywhere
    site = web.TCPSite(runner, host='0.0.0.0', port=port)
    await site.start()
    
    logger.info(f"🌐 HTTP server running on port {port}")
    return runner

# Main bot function
async def main():
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    if not TOKEN:
        logger.error("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        return
    
    bot = MarketSentimentBot(TOKEN)
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CallbackQueryHandler(bot.handle_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    logger.info("🚀 Market Sentiment Bot is starting...")
    
    # Start HTTP server first and wait for it to be ready
    http_runner = await start_http_server()
    
    try:
        # Run the bot
        await application.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        # Clean up the HTTP server when shutting down
        await http_runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
