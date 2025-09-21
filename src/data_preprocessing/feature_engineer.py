import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import ta  # Technical Analysis library
import logging
from typing import Dict, List, Tuple, Optional
from config import MODEL_CONFIG, CUSTOMER_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_lagged_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Create lagged features for time series analysis"""
    try:
        lagged_df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    lagged_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"Created lagged features for {len(columns)} columns with {len(lags)} lags")
        return lagged_df
        
    except Exception as e:
        logger.error(f"Error creating lagged features: {e}")
        return df

def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """Create rolling statistical features"""
    try:
        rolling_df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    rolling_df[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
                    
                    # Rolling standard deviation
                    rolling_df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                    
                    # Rolling min/max
                    rolling_df[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                    rolling_df[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
                    
                    # Rolling quantiles
                    rolling_df[f'{col}_q25_{window}'] = df[col].rolling(window=window).quantile(0.25)
                    rolling_df[f'{col}_q75_{window}'] = df[col].rolling(window=window).quantile(0.75)
        
        logger.info(f"Created rolling features for {len(columns)} columns with {len(windows)} windows")
        return rolling_df
        
    except Exception as e:
        logger.error(f"Error creating rolling features: {e}")
        return df

def create_technical_indicators(df: pd.DataFrame, price_columns: List[str]) -> pd.DataFrame:
    """Create technical analysis indicators for financial time series"""
    try:
        technical_df = df.copy()
        
        for col in price_columns:
            if col in df.columns:
                series = df[col].dropna()
                
                if len(series) > 50:  # Need sufficient data for technical indicators
                    # RSI (Relative Strength Index)
                    technical_df[f'{col}_RSI_14'] = ta.momentum.rsi(series, window=14)
                    
                    # MACD
                    macd_line, macd_signal, macd_histogram = ta.trend.MACD(series).macd(), ta.trend.MACD(series).macd_signal(), ta.trend.MACD(series).macd_diff()
                    technical_df[f'{col}_MACD'] = macd_line
                    technical_df[f'{col}_MACD_signal'] = macd_signal
                    technical_df[f'{col}_MACD_histogram'] = macd_histogram
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = ta.volatility.BollingerBands(series).bollinger_hband(), ta.volatility.BollingerBands(series).bollinger_mavg(), ta.volatility.BollingerBands(series).bollinger_lband()
                    technical_df[f'{col}_BB_upper'] = bb_upper
                    technical_df[f'{col}_BB_middle'] = bb_middle
                    technical_df[f'{col}_BB_lower'] = bb_lower
                    technical_df[f'{col}_BB_width'] = (bb_upper - bb_lower) / bb_middle
                    
                    # Williams %R
                    technical_df[f'{col}_WillR'] = ta.momentum.williams_r(series, series, series)
        
        logger.info(f"Created technical indicators for {len(price_columns)} columns")
        return technical_df
        
    except Exception as e:
        logger.error(f"Error creating technical indicators: {e}")
        return df

def create_rate_change_features(df: pd.DataFrame, rate_columns: List[str]) -> pd.DataFrame:
    """Create interest rate change and momentum features"""
    try:
        rate_df = df.copy()
        
        for col in rate_columns:
            if col in df.columns:
                # First difference (daily change)
                rate_df[f'{col}_change_1d'] = df[col].diff(1)
                
                # Weekly and monthly changes
                rate_df[f'{col}_change_5d'] = df[col].diff(5)
                rate_df[f'{col}_change_21d'] = df[col].diff(21)
                
                # Rate momentum (acceleration)
                rate_df[f'{col}_momentum'] = rate_df[f'{col}_change_1d'].diff(1)
                
                # Volatility (rolling standard deviation of changes)
                rate_df[f'{col}_volatility_5d'] = rate_df[f'{col}_change_1d'].rolling(5).std()
                rate_df[f'{col}_volatility_21d'] = rate_df[f'{col}_change_1d'].rolling(21).std()
                
                # Rate percentile ranking (relative to recent history)
                rate_df[f'{col}_percentile_63d'] = df[col].rolling(63).rank(pct=True)
                rate_df[f'{col}_percentile_252d'] = df[col].rolling(252).rank(pct=True)
        
        logger.info(f"Created rate change features for {len(rate_columns)} columns")
        return rate_df
        
    except Exception as e:
        logger.error(f"Error creating rate change features: {e}")
        return df

def create_yield_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create yield curve shape and dynamics features"""
    try:
        curve_df = df.copy()
        
        # Define yield curve points (if available)
        yield_columns = [col for col in df.columns if 'TREASURY' in col or 'YIELD' in col]
        
        if len(yield_columns) >= 2:
            # Curve steepness and curvature
            if 'TREASURY_10Y' in df.columns and 'TREASURY_2Y' in df.columns:
                curve_df['YIELD_CURVE_STEEPNESS'] = df['TREASURY_10Y'] - df['TREASURY_2Y']
                curve_df['YIELD_CURVE_STEEPNESS_change'] = curve_df['YIELD_CURVE_STEEPNESS'].diff()
            
            # Short end steepness
            if 'TREASURY_2Y' in df.columns and 'TREASURY_3M' in df.columns:
                curve_df['SHORT_END_STEEPNESS'] = df['TREASURY_2Y'] - df['TREASURY_3M']
            
            # Curvature (butterfly)
            if all(col in df.columns for col in ['TREASURY_10Y', 'TREASURY_5Y', 'TREASURY_2Y']):
                curve_df['YIELD_CURVE_CURVATURE'] = (
                    df['TREASURY_2Y'] + df['TREASURY_10Y'] - 2 * df['TREASURY_5Y']
                )
            
            # Level, slope, and curvature from PCA (if enough yield points)
            if len(yield_columns) >= 3:
                yield_data = df[yield_columns].dropna()
                if len(yield_data) > 10:
                    pca = PCA(n_components=3)
                    pca_components = pca.fit_transform(yield_data)
                    
                    # Align with original dataframe
                    pca_df = pd.DataFrame(
                        pca_components,
                        columns=['YIELD_LEVEL', 'YIELD_SLOPE', 'YIELD_CURVATURE'],
                        index=yield_data.index
                    )
                    curve_df = curve_df.join(pca_df, how='left')
        
        logger.info("Created yield curve features")
        return curve_df
        
    except Exception as e:
        logger.error(f"Error creating yield curve features: {e}")
        return df

def create_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create market regime classification features"""
    try:
        regime_df = df.copy()
        
        # Volatility regimes based on VIX
        if 'VIX' in df.columns:
            vix_rolling_mean = df['VIX'].rolling(21).mean()
            vix_rolling_std = df['VIX'].rolling(21).std()
            
            regime_df['VIX_regime_low'] = (df['VIX'] < (vix_rolling_mean - 0.5 * vix_rolling_std)).astype(int)
            regime_df['VIX_regime_high'] = (df['VIX'] > (vix_rolling_mean + 0.5 * vix_rolling_std)).astype(int)
            regime_df['VIX_regime_normal'] = (
                ~regime_df['VIX_regime_low'].astype(bool) & 
                ~regime_df['VIX_regime_high'].astype(bool)
            ).astype(int)
        
        # Interest rate regimes
        if 'FED_FUNDS_RATE' in df.columns:
            fed_rate = df['FED_FUNDS_RATE']
            regime_df['RATE_regime_zero'] = (fed_rate <= 0.25).astype(int)
            regime_df['RATE_regime_low'] = ((fed_rate > 0.25) & (fed_rate <= 2.0)).astype(int)
            regime_df['RATE_regime_normal'] = ((fed_rate > 2.0) & (fed_rate <= 5.0)).astype(int)
            regime_df['RATE_regime_high'] = (fed_rate > 5.0).astype(int)
            
            # Rate cycle indicators (based on recent trend)
            rate_trend_21d = fed_rate.rolling(21).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 21 else 0)
            regime_df['RATE_cycle_tightening'] = (rate_trend_21d > 0.01).astype(int)
            regime_df['RATE_cycle_easing'] = (rate_trend_21d < -0.01).astype(int)
            regime_df['RATE_cycle_stable'] = (
                (rate_trend_21d >= -0.01) & (rate_trend_21d <= 0.01)
            ).astype(int)
        
        # Economic cycle indicators
        if 'UNEMPLOYMENT' in df.columns:
            unemployment_ma = df['UNEMPLOYMENT'].rolling(63).mean()  # 3-month MA
            unemployment_trend = df['UNEMPLOYMENT'] - unemployment_ma
            regime_df['ECON_cycle_recession'] = (unemployment_trend > 0.5).astype(int)
            regime_df['ECON_cycle_recovery'] = (unemployment_trend < -0.5).astype(int)
        
        logger.info("Created market regime features")
        return regime_df
        
    except Exception as e:
        logger.error(f"Error creating market regime features: {e}")
        return df

def create_customer_behavioral_features(customer_df: pd.DataFrame, flows_df: pd.DataFrame) -> pd.DataFrame:
    """Create behavioral features for customer segments"""
    try:
        # Aggregate customer behavior by segment and date
        behavioral_features = flows_df.groupby(['date', 'segment']).agg({
            'deposit_flow': ['sum', 'mean', 'std', 'count'],
            'inflow': 'sum',
            'outflow': 'sum'
        }).round(4)
        
        # Flatten column names
        behavioral_features.columns = [f'segment_{col[1]}_{col[0]}' for col in behavioral_features.columns]
        behavioral_features = behavioral_features.reset_index()
        
        # Calculate segment-specific metrics
        behavioral_features['segment_net_flow_rate'] = (
            behavioral_features['segment_sum_deposit_flow'] / 
            (behavioral_features['segment_sum_inflow'] + behavioral_features['segment_sum_outflow'].abs() + 1)
        )
        
        behavioral_features['segment_inflow_outflow_ratio'] = (
            behavioral_features['segment_sum_inflow'] / 
            (behavioral_features['segment_sum_outflow'].abs() + 1)
        )
        
        # Stability metrics
        behavioral_features['segment_flow_volatility'] = (
            behavioral_features['segment_std_deposit_flow'] / 
            (behavioral_features['segment_mean_deposit_flow'].abs() + 1)
        )
        
        # Create rolling features for behavioral data
        segments = behavioral_features['segment'].unique()
        rolling_behavioral = []
        
        for segment in segments:
            segment_data = behavioral_features[behavioral_features['segment'] == segment].copy()
            segment_data = segment_data.sort_values('date').set_index('date')
            
            # Rolling averages for behavioral metrics
            for window in [7, 21, 63]:
                segment_data[f'segment_flow_ma_{window}'] = segment_data['segment_sum_deposit_flow'].rolling(window).mean()
                segment_data[f'segment_volatility_ma_{window}'] = segment_data['segment_flow_volatility'].rolling(window).mean()
                segment_data[f'segment_inout_ratio_ma_{window}'] = segment_data['segment_inflow_outflow_ratio'].rolling(window).mean()
            
            segment_data['segment'] = segment
            segment_data = segment_data.reset_index()
            rolling_behavioral.append(segment_data)
        
        if rolling_behavioral:
            behavioral_features = pd.concat(rolling_behavioral, ignore_index=True)
        
        logger.info(f"Created behavioral features for {len(segments)} customer segments")
        return behavioral_features
        
    except Exception as e:
        logger.error(f"Error creating customer behavioral features: {e}")
        return pd.DataFrame()

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between market and behavioral variables"""
    try:
        interaction_df = df.copy()
        
        # Rate environment interactions
        if all(col in df.columns for col in ['FED_FUNDS_RATE', 'VIX']):
            interaction_df['RATE_VIX_interaction'] = df['FED_FUNDS_RATE'] * df['VIX']
            interaction_df['RATE_VIX_ratio'] = df['FED_FUNDS_RATE'] / (df['VIX'] + 1)
        
        # Yield curve and volatility interactions
        if all(col in df.columns for col in ['YIELD_CURVE_STEEPNESS', 'VIX']):
            interaction_df['STEEPNESS_VIX_interaction'] = df['YIELD_CURVE_STEEPNESS'] * df['VIX']
        
        # Segment behavior and market condition interactions
        segment_flow_cols = [col for col in df.columns if 'segment_sum_deposit_flow' in col]
        rate_cols = [col for col in df.columns if 'FED_FUNDS_RATE' in col]
        
        for segment_col in segment_flow_cols:
            for rate_col in rate_cols[:2]:  # Limit to avoid too many features
                if segment_col in df.columns and rate_col in df.columns:
                    interaction_name = f'{segment_col}_X_{rate_col}'
                    interaction_df[interaction_name] = df[segment_col] * df[rate_col]
        
        # Volatility and flow interactions
        if 'VIX' in df.columns:
            for segment_col in segment_flow_cols[:3]:  # Limit interactions
                if segment_col in df.columns:
                    interaction_name = f'{segment_col}_X_VIX'
                    interaction_df[interaction_name] = df[segment_col] * df['VIX']
        
        logger.info("Created interaction features")
        return interaction_df
        
    except Exception as e:
        logger.error(f"Error creating interaction features: {e}")
        return df

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features"""
    try:
        calendar_df = df.copy()
        
        if df.index.name != 'date' and 'date' in df.columns:
            date_col = pd.to_datetime(df['date'])
        else:
            date_col = pd.to_datetime(df.index)
        
        # Basic calendar features
        calendar_df['year'] = date_col.year
        calendar_df['month'] = date_col.month
        calendar_df['day'] = date_col.day
        calendar_df['dayofweek'] = date_col.dayofweek
        calendar_df['dayofyear'] = date_col.dayofyear
        calendar_df['week'] = date_col.isocalendar().week
        calendar_df['quarter'] = date_col.quarter
        
        # Cyclical encoding for better ML performance
        calendar_df['month_sin'] = np.sin(2 * np.pi * calendar_df['month'] / 12)
        calendar_df['month_cos'] = np.cos(2 * np.pi * calendar_df['month'] / 12)
        calendar_df['dayofweek_sin'] = np.sin(2 * np.pi * calendar_df['dayofweek'] / 7)
        calendar_df['dayofweek_cos'] = np.cos(2 * np.pi * calendar_df['dayofweek'] / 7)
        calendar_df['dayofyear_sin'] = np.sin(2 * np.pi * calendar_df['dayofyear'] / 365)
        calendar_df['dayofyear_cos'] = np.cos(2 * np.pi * calendar_df['dayofyear'] / 365)
        
        # Business day indicators
        calendar_df['is_weekday'] = (calendar_df['dayofweek'] < 5).astype(int)
        calendar_df['is_weekend'] = (calendar_df['dayofweek'] >= 5).astype(int)
        calendar_df['is_month_start'] = (calendar_df['day'] <= 3).astype(int)
        calendar_df['is_month_end'] = (calendar_df['day'] >= 28).astype(int)
        calendar_df['is_quarter_start'] = ((calendar_df['month'] % 3 == 1) & (calendar_df['day'] <= 3)).astype(int)
        calendar_df['is_quarter_end'] = ((calendar_df['month'] % 3 == 0) & (calendar_df['day'] >= 28)).astype(int)
        calendar_df['is_year_start'] = ((calendar_df['month'] == 1) & (calendar_df['day'] <= 3)).astype(int)
        calendar_df['is_year_end'] = ((calendar_df['month'] == 12) & (calendar_df['day'] >= 28)).astype(int)
        
        # Holiday approximations (US banking holidays)
        calendar_df['is_jan'] = (calendar_df['month'] == 1).astype(int)  # New Year period
        calendar_df['is_dec'] = (calendar_df['month'] == 12).astype(int)  # Holiday season
        calendar_df['is_tax_season'] = ((calendar_df['month'] >= 3) & (calendar_df['month'] <= 4)).astype(int)
        
        logger.info("Created calendar features")
        return calendar_df
        
    except Exception as e:
        logger.error(f"Error creating calendar features: {e}")
        return df

def create_target_variables(flows_df: pd.DataFrame, forecast_horizons: List[int] = [1, 5, 21]) -> pd.DataFrame:
    """Create target variables for different forecast horizons"""
    try:
        target_df = flows_df.copy()
        
        # Sort by date and segment for proper shifting
        target_df = target_df.sort_values(['segment', 'date'])
        
        for horizon in forecast_horizons:
            # Future deposit flows (target for prediction)
            target_df[f'target_deposit_flow_{horizon}d'] = (
                target_df.groupby('segment')['segment_sum_deposit_flow']
                .shift(-horizon)
            )
            
            # Future flow volatility
            target_df[f'target_flow_volatility_{horizon}d'] = (
                target_df.groupby('segment')['segment_flow_volatility']
                .shift(-horizon)
            )
            
            # Future net flow rate
            target_df[f'target_net_flow_rate_{horizon}d'] = (
                target_df.groupby('segment')['segment_net_flow_rate']
                .shift(-horizon)
            )
            
            # Binary targets for significant flows
            flow_threshold = target_df['segment_sum_deposit_flow'].std()
            target_df[f'target_large_outflow_{horizon}d'] = (
                target_df[f'target_deposit_flow_{horizon}d'] < -2 * flow_threshold
            ).astype(int)
            
            target_df[f'target_large_inflow_{horizon}d'] = (
                target_df[f'target_deposit_flow_{horizon}d'] > 2 * flow_threshold
            ).astype(int)
        
        logger.info(f"Created target variables for horizons: {forecast_horizons}")
        return target_df
        
    except Exception as e:
        logger.error(f"Error creating target variables: {e}")
        return flows_df

def engineer_lstm_sequences(df: pd.DataFrame, sequence_length: int = 30, 
                           target_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Engineer sequences for LSTM models"""
    try:
        if target_cols is None:
            target_cols = [col for col in df.columns if col.startswith('target_')]
        
        # Select feature columns (exclude targets and identifiers)
        feature_cols = [
            col for col in df.columns 
            if not col.startswith('target_') and 
               col not in ['date', 'segment', 'customer_id']
        ]
        
        # Prepare data by segment
        X_sequences = []
        y_sequences = []
        
        for segment in df['segment'].unique():
            segment_data = df[df['segment'] == segment].sort_values('date')
            
            if len(segment_data) > sequence_length:
                # Features and targets
                X_segment = segment_data[feature_cols].values
                y_segment = segment_data[target_cols].values if target_cols else None
                
                # Create sequences
                for i in range(sequence_length, len(X_segment)):
                    X_sequences.append(X_segment[i-sequence_length:i])
                    if y_segment is not None:
                        y_sequences.append(y_segment[i])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences) if y_sequences else None
        
        logger.info(f"Created LSTM sequences: X shape {X.shape}, y shape {y.shape if y is not None else 'None'}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error engineering LSTM sequences: {e}")
        return np.array([]), np.array([])

def scale_features(df: pd.DataFrame, target_cols: List[str] = None, 
                  fit_scaler: bool = True, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale features for machine learning models"""
    try:
        if target_cols is None:
            target_cols = [col for col in df.columns if col.startswith('target_')]
        
        # Identify numeric columns to scale
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target variables and identifiers from scaling
        exclude_cols = target_cols + ['customer_id', 'segment']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        scaled_df = df.copy()
        
        if fit_scaler or scaler is None:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(df[feature_cols])
        else:
            scaled_values = scaler.transform(df[feature_cols])
        
        # Update scaled columns
        scaled_df[feature_cols] = scaled_values
        
        logger.info(f"Scaled {len(feature_cols)} feature columns")
        return scaled_df, scaler
        
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        return df, StandardScaler()

def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """Handle missing values in the dataset"""
    try:
        clean_df = df.copy()
        initial_missing = clean_df.isnull().sum().sum()
        
        if method == 'forward_fill':
            # Forward fill then backward fill
            clean_df = clean_df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            # Linear interpolation for numeric columns
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
            clean_df[numeric_cols] = clean_df[numeric_cols].interpolate(method='linear')
            clean_df = clean_df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'zero_fill':
            clean_df = clean_df.fillna(0)
        elif method == 'mean_fill':
            # Fill with column means for numeric, mode for categorical
            for col in clean_df.columns:
                if clean_df[col].dtype in [np.number]:
                    clean_df[col].fillna(clean_df[col].mean(), inplace=True)
                else:
                    clean_df[col].fillna(clean_df[col].mode().iloc[0] if not clean_df[col].mode().empty else 'Unknown', inplace=True)
        
        final_missing = clean_df.isnull().sum().sum()
        logger.info(f"Handled missing values: {initial_missing} -> {final_missing}")
        
        return clean_df
        
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        return df

def select_features(df: pd.DataFrame, target_col: str, method: str = 'correlation', 
                   max_features: int = 50) -> List[str]:
    """Select most important features for modeling"""
    try:
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return []
        
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('target_')]
        
        if method == 'correlation':
            # Correlation-based feature selection
            correlations = df[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            selected_features = correlations.head(max_features).index.tolist()
            selected_features = [col for col in selected_features if col != target_col]
            
        elif method == 'variance':
            # Variance-based selection (remove low variance features)
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(df[feature_cols])
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.variances_[i] > 0.01]
            selected_features = selected_features[:max_features]
            
        else:
            selected_features = feature_cols[:max_features]
        
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        return selected_features
        
    except Exception as e:
        logger.error(f"Error selecting features: {e}")
        return []

def create_feature_engineering_pipeline(market_data: pd.DataFrame, customer_data: pd.DataFrame, 
                                       flows_data: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering pipeline"""
    logger.info("=== Starting Feature Engineering Pipeline ===")
    
    try:
        # 1. Create market-based features
        logger.info("Step 1: Creating market features...")
        market_features = market_data.copy()
        
        # Rate columns for feature engineering
        rate_columns = [col for col in market_data.columns if 'RATE' in col or 'TREASURY' in col or 'YIELD' in col]
        price_columns = [col for col in market_data.columns if any(x in col for x in ['TREASURY', 'VIX', 'EUR', 'GBP'])]
        
        # Create lagged features
        market_features = create_lagged_features(market_features, rate_columns, [1, 5, 21])
        
        # Create rolling features
        market_features = create_rolling_features(market_features, rate_columns + price_columns, [5, 21, 63])
        
        # Create technical indicators
        market_features = create_technical_indicators(market_features, price_columns)
        
        # Create rate change features
        market_features = create_rate_change_features(market_features, rate_columns)
        
        # Create yield curve features
        market_features = create_yield_curve_features(market_features)
        
        # Create market regime features
        market_features = create_market_regime_features(market_features)
        
        # Add calendar features
        market_features = create_calendar_features(market_features)
        
        logger.info(f"Market features shape: {market_features.shape}")
        
        # 2. Create customer behavioral features
        logger.info("Step 2: Creating behavioral features...")
        behavioral_features = create_customer_behavioral_features(customer_data, flows_data)
        
        logger.info(f"Behavioral features shape: {behavioral_features.shape}")
        
        # 3. Merge market and behavioral features
        logger.info("Step 3: Merging features...")
        if 'date' in market_features.columns:
            market_features['date'] = pd.to_datetime(market_features['date'])
        else:
            market_features = market_features.reset_index()
            market_features['date'] = pd.to_datetime(market_features['date'])
            
        behavioral_features['date'] = pd.to_datetime(behavioral_features['date'])
        
        # Merge on date
        combined_features = behavioral_features.merge(market_features, on='date', how='left')
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        
        # 4. Create interaction features
        logger.info("Step 4: Creating interaction features...")
        combined_features = create_interaction_features(combined_features)
        
        # 5. Create target variables
        logger.info("Step 5: Creating target variables...")
        combined_features = create_target_variables(combined_features, [1, 5, 21])
        
        # 6. Handle missing values
        logger.info("Step 6: Handling missing values...")
        combined_features = handle_missing_values(combined_features, method='forward_fill')
        
        # 7. Remove rows with insufficient data for targets
        logger.info("Step 7: Cleaning final dataset...")
        # Remove last few rows that don't have target values
        target_cols = [col for col in combined_features.columns if col.startswith('target_')]
        combined_features = combined_features.dropna(subset=target_cols)
        
        logger.info(f"Final feature dataset shape: {combined_features.shape}")
        logger.info(f"Features: {len([col for col in combined_features.columns if not col.startswith('target_')])}")
        logger.info(f"Targets: {len(target_cols)}")
        
        # Save feature-engineered dataset
        combined_features.to_csv('data/processed/feature_engineered_dataset.csv', index=False)
        logger.info("=== Feature Engineering Pipeline Complete ===")
        
        return combined_features
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        return pd.DataFrame()

# Utility functions for model preparation
def prepare_model_datasets(df: pd.DataFrame, target_col: str, 
                          test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare train/test splits for modeling"""
    try:
        # Sort by date to ensure temporal split
        df_sorted = df.sort_values('date')
        
        # Calculate split point
        split_point = int(len(df_sorted) * (1 - test_size))
        
        # Split data
        train_df = df_sorted.iloc[:split_point].copy()
        test_df = df_sorted.iloc[split_point:].copy()
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col not in ['date', 'segment']]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error preparing model datasets: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

if __name__ == "__main__":
    # Example usage
    logger.info("Feature engineering module loaded successfully")
    print("Available functions:")
    print("- create_lagged_features")
    print("- create_rolling_features") 
    print("- create_technical_indicators")
    print("- create_rate_change_features")
    print("- create_yield_curve_features")
    print("- create_market_regime_features")
    print("- create_customer_behavioral_features")
    print("- create_interaction_features")
    print("- create_calendar_features")
    print("- create_target_variables")
    print("- create_feature_engineering_pipeline")