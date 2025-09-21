import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_CONFIG = {
    'FRED_API_KEY': os.getenv('FRED_API_KEY', 'your_fred_api_key_here'),
    'ECB_API_BASE': 'https://sdw-wsrest.ecb.europa.eu/service/data',
    'TREASURY_API_BASE': 'https://api.fiscaldata.treasury.gov/services/api/v1',
    'RATE_LIMIT_DELAY': 1.0,  # seconds between API calls
    'MAX_RETRIES': 3
}

# Data Configuration
DATA_CONFIG = {
    'START_DATE': (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d'),  # 5 years
    'END_DATE': datetime.now().strftime('%Y-%m-%d'),
    'FRED_SERIES': {
        'FED_FUNDS_RATE': 'FEDFUNDS',
        'GDP_GROWTH': 'GDP',
        'UNEMPLOYMENT': 'UNRATE',
        'INFLATION': 'CPIAUCSL',
        'TREASURY_10Y': 'GS10',
        'TREASURY_2Y': 'GS2',
        'TREASURY_3M': 'GS3M',
        'VIX': 'VIXCLS',
        'EUR_USD': 'DEXUSEU',
        'LIBOR_3M': 'USD3MTD156N'
    },
    'ECB_SERIES': {
        'ECB_MAIN_RATE': 'IRS/M/DE+FR+IT+ES+NL/L/A/MIR_MFI_RT/A/2240',
        'EURIBOR_3M': 'FM/B/U2/EUR/MM/EURIBOR3MD/HSTA',
        'GERMAN_BUND_10Y': 'IRS/M/DE/L/A/A/MIR_IR_LTGBY/A/2300'
    }
}

# Model Configuration
MODEL_CONFIG = {
    'RANDOM_FOREST': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'XGBOOST': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    },
    'LSTM': {
        'units': 50,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'lookback_days': 30
    },
    'ENSEMBLE_WEIGHTS': {
        'random_forest': 0.3,
        'xgboost': 0.4,
        'lstm': 0.3
    }
}

# Customer Segmentation Configuration
CUSTOMER_CONFIG = {
    'SEGMENTS': {
        'RETAIL_SMALL': {'min_balance': 0, 'max_balance': 10000},
        'RETAIL_MEDIUM': {'min_balance': 10000, 'max_balance': 100000},
        'RETAIL_LARGE': {'min_balance': 100000, 'max_balance': 1000000},
        'SME': {'min_balance': 50000, 'max_balance': 5000000},
        'CORPORATE': {'min_balance': 1000000, 'max_balance': float('inf')}
    },
    'BEHAVIORAL_PARAMS': {
        'RATE_SENSITIVITY': {'low': 0.1, 'medium': 0.3, 'high': 0.7},
        'VOLATILITY': {'low': 0.05, 'medium': 0.15, 'high': 0.3},
        'LOYALTY': {'low': 0.3, 'medium': 0.6, 'high': 0.9}
    }
}

# Risk Configuration
RISK_CONFIG = {
    'LCR_MINIMUM': 1.0,  # 100% minimum
    'LCR_BUFFER': 0.1,   # 10% buffer
    'STRESS_SCENARIOS': {
        'MILD': {'rate_shock': 0.01, 'deposit_outflow': 0.1},
        'MODERATE': {'rate_shock': 0.025, 'deposit_outflow': 0.2},
        'SEVERE': {'rate_shock': 0.05, 'deposit_outflow': 0.4}
    },
    'VAR_CONFIDENCE': 0.95,
    'BACKTESTING_PERIOD': 252  # trading days
}

# Visualization Configuration
VIZ_CONFIG = {
    'COLORS': {
        'PRIMARY': '#1f77b4',
        'SECONDARY': '#ff7f0e',
        'SUCCESS': '#2ca02c',
        'WARNING': '#ffbb78',
        'DANGER': '#d62728',
        'GRADIENT': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    },
    'CHART_HEIGHT': 400,
    'CHART_WIDTH': 800,
    'DPI': 100
}

# File Paths
PATHS = {
    'DATA_RAW': 'data/raw/',
    'DATA_PROCESSED': 'data/processed/',
    'MODELS': 'data/models/',
    'LOGS': 'logs/',
    'OUTPUTS': 'outputs/'
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'PAGE_TITLE': 'Treasury Behavioral Deposit Analytics',
    'PAGE_ICON': '🏦',
    'LAYOUT': 'wide',
    'SIDEBAR_STATE': 'expanded',
    'REFRESH_INTERVAL': 300  # seconds
}

# Logging Configuration
LOGGING_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_FILE': 'logs/treasury_analytics.log'
}

# Validation thresholds
VALIDATION_CONFIG = {
    'MIN_ACCURACY_THRESHOLD': 0.75,
    'MAX_MAPE_THRESHOLD': 0.15,  # 15% Mean Absolute Percentage Error
    'MIN_DATA_POINTS': 100,
    'CORRELATION_THRESHOLD': 0.7
}