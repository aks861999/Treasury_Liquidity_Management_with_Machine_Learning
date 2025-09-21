import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from fredapi import Fred
import yfinance as yf
import logging
from typing import Dict, List, Optional, Tuple
from config import API_CONFIG, DATA_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_fred_client() -> Fred:
    """Initialize FRED API client"""
    try:
        fred = Fred(api_key=API_CONFIG['FRED_API_KEY'])
        return fred
    except Exception as e:
        logger.error(f"Failed to initialize FRED client: {e}")
        raise

def fetch_fred_data(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch data from FRED API"""
    try:
        fred = initialize_fred_client()
        data = fred.get_series(series_id, start=start_date, end=end_date)
        logger.info(f"Successfully fetched {series_id} data: {len(data)} points")
        return data
    except Exception as e:
        logger.error(f"Error fetching FRED data for {series_id}: {e}")
        return pd.Series()

def fetch_treasury_yields() -> pd.DataFrame:
    """Fetch US Treasury yield curve data"""
    try:
        treasury_data = {}
        for name, series_id in DATA_CONFIG['FRED_SERIES'].items():
            if 'TREASURY' in name or 'GS' in series_id:
                data = fetch_fred_data(
                    series_id, 
                    DATA_CONFIG['START_DATE'], 
                    DATA_CONFIG['END_DATE']
                )
                if not data.empty:
                    treasury_data[name] = data
                time.sleep(API_CONFIG['RATE_LIMIT_DELAY'])
        
        if treasury_data:
            df = pd.DataFrame(treasury_data)
            df.index = pd.to_datetime(df.index)
            return df.fillna(method='ffill')
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching Treasury yields: {e}")
        return pd.DataFrame()

def fetch_economic_indicators() -> pd.DataFrame:
    """Fetch key economic indicators"""
    try:
        economic_data = {}
        indicators = ['FED_FUNDS_RATE', 'GDP_GROWTH', 'UNEMPLOYMENT', 'INFLATION', 'VIX']
        
        for indicator in indicators:
            if indicator in DATA_CONFIG['FRED_SERIES']:
                series_id = DATA_CONFIG['FRED_SERIES'][indicator]
                data = fetch_fred_data(
                    series_id,
                    DATA_CONFIG['START_DATE'],
                    DATA_CONFIG['END_DATE']
                )
                if not data.empty:
                    economic_data[indicator] = data
                time.sleep(API_CONFIG['RATE_LIMIT_DELAY'])
        
        if economic_data:
            df = pd.DataFrame(economic_data)
            df.index = pd.to_datetime(df.index)
            return df.fillna(method='ffill')
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
        return pd.DataFrame()

def fetch_fx_rates() -> pd.DataFrame:
    """Fetch foreign exchange rates"""
    try:
        fx_symbols = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'CHFUSD=X']
        fx_data = {}
        
        for symbol in fx_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=DATA_CONFIG['START_DATE'],
                    end=DATA_CONFIG['END_DATE']
                )
                if not hist.empty:
                    fx_data[symbol.replace('=X', '')] = hist['Close']
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue
        
        if fx_data:
            df = pd.DataFrame(fx_data)
            return df.fillna(method='ffill')
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching FX rates: {e}")
        return pd.DataFrame()

def calculate_yield_curve_features(treasury_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate yield curve derived features"""
    try:
        features = pd.DataFrame(index=treasury_df.index)
        
        # Yield spreads
        if 'TREASURY_10Y' in treasury_df.columns and 'TREASURY_2Y' in treasury_df.columns:
            features['YIELD_SPREAD_10Y2Y'] = (
                treasury_df['TREASURY_10Y'] - treasury_df['TREASURY_2Y']
            )
        
        if 'TREASURY_2Y' in treasury_df.columns and 'TREASURY_3M' in treasury_df.columns:
            features['YIELD_SPREAD_2Y3M'] = (
                treasury_df['TREASURY_2Y'] - treasury_df['TREASURY_3M']
            )
        
        # Yield curve slope (10Y - 3M)
        if 'TREASURY_10Y' in treasury_df.columns and 'TREASURY_3M' in treasury_df.columns:
            features['YIELD_CURVE_SLOPE'] = (
                treasury_df['TREASURY_10Y'] - treasury_df['TREASURY_3M']
            )
        
        # Rate changes (first difference)
        for col in treasury_df.columns:
            if 'TREASURY' in col:
                features[f'{col}_CHANGE'] = treasury_df[col].diff()
                features[f'{col}_VOLATILITY'] = (
                    treasury_df[col].rolling(window=30).std()
                )
        
        return features.fillna(0)
        
    except Exception as e:
        logger.error(f"Error calculating yield curve features: {e}")
        return pd.DataFrame()

def fetch_volatility_indices() -> pd.DataFrame:
    """Fetch market volatility indices"""
    try:
        vol_symbols = ['^VIX', '^VSTOXX']  # US VIX and European VSTOXX
        vol_data = {}
        
        for symbol in vol_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=DATA_CONFIG['START_DATE'],
                    end=DATA_CONFIG['END_DATE']
                )
                if not hist.empty:
                    vol_data[symbol.replace('^', '')] = hist['Close']
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue
        
        if vol_data:
            df = pd.DataFrame(vol_data)
            return df.fillna(method='ffill')
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching volatility indices: {e}")
        return pd.DataFrame()

def create_market_regime_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Create market regime indicators based on volatility and rates"""
    try:
        regime_df = pd.DataFrame(index=data.index)
        
        # Interest rate regime (based on fed funds rate)
        if 'FED_FUNDS_RATE' in data.columns:
            fed_rate = data['FED_FUNDS_RATE']
            regime_df['RATE_REGIME_LOW'] = (fed_rate <= 2.0).astype(int)
            regime_df['RATE_REGIME_NORMAL'] = ((fed_rate > 2.0) & (fed_rate <= 5.0)).astype(int)
            regime_df['RATE_REGIME_HIGH'] = (fed_rate > 5.0).astype(int)
        
        # Volatility regime (based on VIX if available)
        if 'VIX' in data.columns:
            vix = data['VIX']
            regime_df['VOL_REGIME_LOW'] = (vix <= 15).astype(int)
            regime_df['VOL_REGIME_NORMAL'] = ((vix > 15) & (vix <= 25)).astype(int)
            regime_df['VOL_REGIME_HIGH'] = (vix > 25).astype(int)
        
        # Yield curve regime
        if 'YIELD_CURVE_SLOPE' in data.columns:
            slope = data['YIELD_CURVE_SLOPE']
            regime_df['CURVE_INVERTED'] = (slope < 0).astype(int)
            regime_df['CURVE_FLAT'] = ((slope >= 0) & (slope <= 1.0)).astype(int)
            regime_df['CURVE_STEEP'] = (slope > 1.0).astype(int)
        
        return regime_df
        
    except Exception as e:
        logger.error(f"Error creating market regime indicators: {e}")
        return pd.DataFrame()

def aggregate_market_data() -> pd.DataFrame:
    """Aggregate all market data into a single DataFrame"""
    try:
        logger.info("Starting market data collection...")
        
        # Fetch all data sources
        treasury_data = fetch_treasury_yields()
        economic_data = fetch_economic_indicators()
        fx_data = fetch_fx_rates()
        vol_data = fetch_volatility_indices()
        
        # Combine all data
        market_data = pd.DataFrame()
        
        for df, name in [
            (treasury_data, "Treasury"),
            (economic_data, "Economic"),
            (fx_data, "FX"),
            (vol_data, "Volatility")
        ]:
            if not df.empty:
                if market_data.empty:
                    market_data = df.copy()
                else:
                    market_data = market_data.join(df, how='outer')
                logger.info(f"Added {name} data: {df.shape}")
        
        if market_data.empty:
            logger.warning("No market data collected")
            return pd.DataFrame()
        
        # Calculate derived features
        yield_features = calculate_yield_curve_features(treasury_data)
        if not yield_features.empty:
            market_data = market_data.join(yield_features, how='outer')
        
        # Add market regime indicators
        regime_features = create_market_regime_indicators(market_data)
        if not regime_features.empty:
            market_data = market_data.join(regime_features, how='outer')
        
        # Forward fill missing values
        market_data = market_data.fillna(method='ffill')
        
        # Add time-based features
        market_data['MONTH'] = market_data.index.month
        market_data['QUARTER'] = market_data.index.quarter
        market_data['DAY_OF_WEEK'] = market_data.index.dayofweek
        market_data['WEEK_OF_YEAR'] = market_data.index.isocalendar().week
        
        logger.info(f"Final market data shape: {market_data.shape}")
        return market_data
        
    except Exception as e:
        logger.error(f"Error aggregating market data: {e}")
        return pd.DataFrame()

def save_market_data(data: pd.DataFrame, filepath: str) -> bool:
    """Save market data to CSV file"""
    try:
        data.to_csv(filepath)
        logger.info(f"Market data saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving market data: {e}")
        return False

# Main execution function
def collect_market_data() -> pd.DataFrame:
    """Main function to collect and process all market data"""
    logger.info("=== Starting Market Data Collection ===")
    
    market_data = aggregate_market_data()
    
    if not market_data.empty:
        # Save to file
        filepath = f"{DATA_CONFIG['START_DATE']}_to_{DATA_CONFIG['END_DATE']}_market_data.csv"
        save_market_data(market_data, f"data/raw/{filepath}")
        
        logger.info("=== Market Data Collection Complete ===")
        return market_data
    else:
        logger.error("Failed to collect market data")
        return pd.DataFrame()

if __name__ == "__main__":
    data = collect_market_data()
    print(f"Collected market data with shape: {data.shape}")
    if not data.empty:
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")