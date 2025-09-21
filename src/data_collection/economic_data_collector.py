import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from fredapi import Fred
import logging
from typing import Dict, List, Tuple, Optional
from config import API_CONFIG, DATA_CONFIG
from src.utils.api_utils import make_api_request, fetch_fred_data, APIRateLimiter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_economic_indicators() -> pd.DataFrame:
    """Collect comprehensive economic indicators"""
    try:
        logger.info("Starting economic indicators collection...")
        
        # Economic indicators to collect
        economic_series = {
            'GDP_REAL': 'GDPC1',
            'GDP_NOMINAL': 'GDP',
            'GDP_GROWTH': 'A191RL1Q225SBEA',
            'UNEMPLOYMENT_RATE': 'UNRATE',
            'INFLATION_CPI': 'CPIAUCSL',
            'INFLATION_CORE': 'CPILFESL',
            'INFLATION_PCE': 'PCEPI',
            'INDUSTRIAL_PRODUCTION': 'INDPRO',
            'CONSUMER_SENTIMENT': 'UMCSENT',
            'RETAIL_SALES': 'RSAFS',
            'HOUSING_STARTS': 'HOUST',
            'EMPLOYMENT_NONFARM': 'PAYEMS',
            'WAGES_HOURLY': 'AHETPI',
            'PRODUCTIVITY': 'PRS85006092',
            'TRADE_BALANCE': 'BOPGSTB'
        }
        
        # Initialize rate limiter
        rate_limiter = APIRateLimiter(max_calls=120, time_window=60)
        fred_client = Fred(api_key=API_CONFIG['FRED_API_KEY'])
        
        economic_data = {}
        
        for indicator_name, series_id in economic_series.items():
            try:
                rate_limiter.wait_if_needed()
                
                series_data = fred_client.get_series(
                    series_id,
                    start=DATA_CONFIG['START_DATE'],
                    end=DATA_CONFIG['END_DATE']
                )
                
                if not series_data.empty:
                    economic_data[indicator_name] = series_data
                    logger.info(f"Collected {indicator_name}: {len(series_data)} observations")
                else:
                    logger.warning(f"No data for {indicator_name} ({series_id})")
                
                time.sleep(API_CONFIG['RATE_LIMIT_DELAY'])
                
            except Exception as e:
                logger.error(f"Error collecting {indicator_name}: {e}")
                continue
        
        if economic_data:
            # Combine into DataFrame
            econ_df = pd.DataFrame(economic_data)
            econ_df.index = pd.to_datetime(econ_df.index)
            
            # Forward fill missing values
            econ_df = econ_df.fillna(method='ffill')
            
            logger.info(f"Economic indicators collection completed: {econ_df.shape}")
            return econ_df
        else:
            logger.warning("No economic data collected")
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error in economic indicators collection: {e}")
        return pd.DataFrame()

def collect_labor_market_data() -> pd.DataFrame:
    """Collect detailed labor market indicators"""
    try:
        logger.info("Collecting labor market data...")
        
        labor_series = {
            'UNEMPLOYMENT_RATE': 'UNRATE',
            'LABOR_FORCE_PARTICIPATION': 'CIVPART',
            'EMPLOYMENT_POPULATION_RATIO': 'EMRATIO',
            'UNEMPLOYMENT_DURATION': 'UEMPMEAN',
            'INITIAL_CLAIMS': 'ICSA',
            'CONTINUING_CLAIMS': 'CCSA',
            'JOB_OPENINGS': 'JTSJOL',
            'QUITS_RATE': 'JTSQUR',
            'HIRES_RATE': 'JTSHIR',
            'LAYOFFS_RATE': 'JTSLDL'
        }
        
        fred_client = Fred(api_key=API_CONFIG['FRED_API_KEY'])
        rate_limiter = APIRateLimiter()
        
        labor_data = {}
        
        for indicator, series_id in labor_series.items():
            try:
                rate_limiter.wait_if_needed()
                
                data = fred_client.get_series(
                    series_id,
                    start=DATA_CONFIG['START_DATE'],
                    end=DATA_CONFIG['END_DATE']
                )
                
                if not data.empty:
                    labor_data[indicator] = data
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to collect {indicator}: {e}")
                continue
        
        if labor_data:
            labor_df = pd.DataFrame(labor_data)
            labor_df.index = pd.to_datetime(labor_df.index)
            labor_df = labor_df.fillna(method='ffill')
            
            logger.info(f"Labor market data collected: {labor_df.shape}")
            return labor_df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error collecting labor market data: {e}")
        return pd.DataFrame()

def collect_inflation_data() -> pd.DataFrame:
    """Collect comprehensive inflation indicators"""
    try:
        logger.info("Collecting inflation data...")
        
        inflation_series = {
            'CPI_ALL': 'CPIAUCSL',
            'CPI_CORE': 'CPILFESL',
            'CPI_ENERGY': 'CPIENGSL',
            'CPI_FOOD': 'CPIUFDSL',
            'CPI_SHELTER': 'CPISHSL',
            'CPI_MEDICAL': 'CPIMEDSL',
            'PCE_ALL': 'PCEPI',
            'PCE_CORE': 'PCEPILFE',
            'PPI_FINAL': 'PPIFIS',
            'PPI_INTERMEDIATE': 'PPIITM',
            'PPI_CRUDE': 'PPICRM',
            'IMPORT_PRICES': 'IR',
            'EXPORT_PRICES': 'IQ',
            'BREAKEVEN_5Y': 'T5YIE',
            'BREAKEVEN_10Y': 'T10YIE'
        }
        
        fred_client = Fred(api_key=API_CONFIG['FRED_API_KEY'])
        rate_limiter = APIRateLimiter()
        
        inflation_data = {}
        
        for indicator, series_id in inflation_series.items():
            try:
                rate_limiter.wait_if_needed()
                
                data = fred_client.get_series(
                    series_id,
                    start=DATA_CONFIG['START_DATE'],
                    end=DATA_CONFIG['END_DATE']
                )
                
                if not data.empty:
                    inflation_data[indicator] = data
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to collect {indicator}: {e}")
                continue
        
        if inflation_data:
            inflation_df = pd.DataFrame(inflation_data)
            inflation_df.index = pd.to_datetime(inflation_df.index)
            
            # Calculate inflation rates (YoY changes)
            for col in ['CPI_ALL', 'CPI_CORE', 'PCE_ALL', 'PCE_CORE']:
                if col in inflation_df.columns:
                    inflation_df[f'{col}_YOY'] = inflation_df[col].pct_change(12) * 100
            
            inflation_df = inflation_df.fillna(method='ffill')
            
            logger.info(f"Inflation data collected: {inflation_df.shape}")
            return inflation_df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error collecting inflation data: {e}")
        return pd.DataFrame()

def collect_business_cycle_indicators() -> pd.DataFrame:
    """Collect business cycle and leading indicators"""
    try:
        logger.info("Collecting business cycle indicators...")
        
        business_series = {
            'LEADING_INDEX': 'USSLIND',
            'COINCIDENT_INDEX': 'USALOLITONOSTSAM',
            'LAGGING_INDEX': 'USLAGHD',
            'RECESSION_INDICATOR': 'USREC',
            'YIELD_CURVE_SPREAD': 'T10Y2Y',
            'CREDIT_SPREAD_BAA': 'BAA10Y',
            'CREDIT_SPREAD_AAA': 'AAA10Y',
            'TERM_SPREAD': 'T10Y3M',
            'CONSUMER_CONFIDENCE': 'CSCICP03USM665S',
            'BUSINESS_CONFIDENCE': 'BSCICP03USM665S',
            'STOCK_MARKET_CAP': 'DDDM01USA156NWDB',
            'CONSUMER_CREDIT': 'TOTALSL',
            'BANK_CREDIT': 'TOTLL',
            'MONEY_SUPPLY_M2': 'M2SL'
        }
        
        fred_client = Fred(api_key=API_CONFIG['FRED_API_KEY'])
        rate_limiter = APIRateLimiter()
        
        business_data = {}
        
        for indicator, series_id in business_series.items():
            try:
                rate_limiter.wait_if_needed()
                
                data = fred_client.get_series(
                    series_id,
                    start=DATA_CONFIG['START_DATE'],
                    end=DATA_CONFIG['END_DATE']
                )
                
                if not data.empty:
                    business_data[indicator] = data
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to collect {indicator}: {e}")
                continue
        
        if business_data:
            business_df = pd.DataFrame(business_data)
            business_df.index = pd.to_datetime(business_df.index)
            business_df = business_df.fillna(method='ffill')
            
            logger.info(f"Business cycle data collected: {business_df.shape}")
            return business_df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error collecting business cycle data: {e}")
        return pd.DataFrame()

def create_economic_indicators_composite() -> pd.DataFrame:
    """Create composite economic indicators dataset"""
    try:
        logger.info("Creating composite economic indicators...")
        
        # Collect all economic data
        economic_data = collect_economic_indicators()
        labor_data = collect_labor_market_data()
        inflation_data = collect_inflation_data()
        business_data = collect_business_cycle_indicators()
        
        # Combine all datasets
        datasets = [economic_data, labor_data, inflation_data, business_data]
        non_empty_datasets = [df for df in datasets if not df.empty]
        
        if not non_empty_datasets:
            logger.error("No economic data collected")
            return pd.DataFrame()
        
        # Start with first dataset
        composite_df = non_empty_datasets[0].copy()
        
        # Join other datasets
        for df in non_empty_datasets[1:]:
            composite_df = composite_df.join(df, how='outer')
        
        # Create derived indicators
        composite_df = create_derived_economic_indicators(composite_df)
        
        # Fill missing values
        composite_df = composite_df.fillna(method='ffill')
        composite_df = composite_df.fillna(method='bfill')
        
        logger.info(f"Composite economic indicators created: {composite_df.shape}")
        return composite_df
        
    except Exception as e:
        logger.error(f"Error creating composite economic indicators: {e}")
        return pd.DataFrame()

def create_derived_economic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived economic indicators from base data"""
    try:
        derived_df = df.copy()
        
        # Economic growth indicators
        if 'GDP_REAL' in df.columns:
            derived_df['GDP_GROWTH_QOQ'] = df['GDP_REAL'].pct_change() * 100
            derived_df['GDP_GROWTH_YOY'] = df['GDP_REAL'].pct_change(4) * 100
        
        # Labor market efficiency
        if all(col in df.columns for col in ['UNEMPLOYMENT_RATE', 'JOB_OPENINGS']):
            derived_df['BEVERIDGE_CURVE'] = df['JOB_OPENINGS'] / df['UNEMPLOYMENT_RATE']
        
        # Inflation momentum
        if 'CPI_ALL' in df.columns:
            derived_df['INFLATION_MOMENTUM'] = df['CPI_ALL'].pct_change(3) * 100  # 3-month change
            derived_df['INFLATION_ACCELERATION'] = derived_df['INFLATION_MOMENTUM'].diff()
        
        # Credit conditions
        if all(col in df.columns for col in ['CREDIT_SPREAD_BAA', 'CREDIT_SPREAD_AAA']):
            derived_df['CREDIT_RISK_SPREAD'] = df['CREDIT_SPREAD_BAA'] - df['CREDIT_SPREAD_AAA']
        
        # Economic stress index
        stress_components = []
        if 'VIX' in df.columns:
            stress_components.append(df['VIX'])
        if 'CREDIT_SPREAD_BAA' in df.columns:
            stress_components.append(df['CREDIT_SPREAD_BAA'])
        if 'UNEMPLOYMENT_RATE' in df.columns:
            stress_components.append(df['UNEMPLOYMENT_RATE'])
        
        if stress_components:
            # Normalize and combine
            normalized_components = []
            for component in stress_components:
                normalized = (component - component.mean()) / component.std()
                normalized_components.append(normalized)
            
            derived_df['ECONOMIC_STRESS_INDEX'] = sum(normalized_components) / len(normalized_components)
        
        # Business cycle phases
        if 'LEADING_INDEX' in df.columns:
            leading_trend = df['LEADING_INDEX'].rolling(6).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 6 else 0)
            derived_df['BUSINESS_CYCLE_PHASE'] = np.where(
                leading_trend > 0.1, 1,  # Expansion
                np.where(leading_trend < -0.1, -1, 0)  # Contraction vs Stable
            )
        
        logger.info("Derived economic indicators created")
        return derived_df
        
    except Exception as e:
        logger.error(f"Error creating derived indicators: {e}")
        return df

def save_economic_data(economic_df: pd.DataFrame, filename: str = None) -> bool:
    """Save economic data to file"""
    try:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/raw/economic_indicators_{timestamp}.csv'
        
        economic_df.to_csv(filename)
        logger.info(f"Economic data saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving economic data: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    logger.info("Starting economic data collection...")
    
    economic_indicators = create_economic_indicators_composite()
    
    if not economic_indicators.empty:
        print(f"Economic indicators collected: {economic_indicators.shape}")
        print(f"Columns: {list(economic_indicators.columns)}")
        print(f"Date range: {economic_indicators.index.min()} to {economic_indicators.index.max()}")
        
        # Save data
        save_economic_data(economic_indicators)
    else:
        print("Failed to collect economic data")