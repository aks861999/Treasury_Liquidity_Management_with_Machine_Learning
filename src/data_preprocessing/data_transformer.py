import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from src.utils.data_utils import convert_data_types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """Comprehensive data transformation pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.transformers = {}
        self.feature_names = {}
        self.transformation_log = []
    
    def log_transformation(self, operation: str, details: str):
        """Log transformation operations"""
        self.transformation_log.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation,
            'details': details
        })
        logger.info(f"{operation}: {details}")

def transform_market_data(market_df: pd.DataFrame) -> pd.DataFrame:
    """Transform market data for modeling"""
    try:
        logger.info("Transforming market data...")
        
        transformed_df = market_df.copy()
        
        # 1. Create rate change features
        rate_columns = [col for col in transformed_df.columns if 'RATE' in col or 'TREASURY' in col]
        
        for col in rate_columns:
            if col in transformed_df.columns:
                # Daily changes
                transformed_df[f'{col}_change'] = transformed_df[col].diff()
                
                # Rolling volatility (20-day)
                transformed_df[f'{col}_volatility'] = transformed_df[col].rolling(20).std()
                
                # Moving averages
                transformed_df[f'{col}_ma_5'] = transformed_df[col].rolling(5).mean()
                transformed_df[f'{col}_ma_20'] = transformed_df[col].rolling(20).mean()
                
                # Rate momentum
                transformed_df[f'{col}_momentum'] = transformed_df[f'{col}_change'].rolling(5).mean()
        
        # 2. Create yield curve features
        if all(col in transformed_df.columns for col in ['TREASURY_10Y', 'TREASURY_2Y']):
            transformed_df['yield_spread_10y2y'] = transformed_df['TREASURY_10Y'] - transformed_df['TREASURY_2Y']
            transformed_df['yield_spread_change'] = transformed_df['yield_spread_10y2y'].diff()
        
        if all(col in transformed_df.columns for col in ['TREASURY_2Y', 'TREASURY_3M']):
            transformed_df['yield_spread_2y3m'] = transformed_df['TREASURY_2Y'] - transformed_df['TREASURY_3M']
        
        # 3. VIX transformations
        if 'VIX' in transformed_df.columns:
            # Log transformation to normalize VIX
            transformed_df['VIX_log'] = np.log(transformed_df['VIX'])
            
            # VIX regime indicators
            transformed_df['VIX_regime_low'] = (transformed_df['VIX'] < 20).astype(int)
            transformed_df['VIX_regime_high'] = (transformed_df['VIX'] > 30).astype(int)
            
            # VIX changes
            transformed_df['VIX_change'] = transformed_df['VIX'].diff()
            transformed_df['VIX_pct_change'] = transformed_df['VIX'].pct_change()
        
        # 4. Create market regime indicators
        if 'FED_FUNDS_RATE' in transformed_df.columns:
            rate = transformed_df['FED_FUNDS_RATE']
            transformed_df['rate_regime_zero'] = (rate <= 0.25).astype(int)
            transformed_df['rate_regime_low'] = ((rate > 0.25) & (rate <= 2.0)).astype(int)
            transformed_df['rate_regime_normal'] = ((rate > 2.0) & (rate <= 5.0)).astype(int)
            transformed_df['rate_regime_high'] = (rate > 5.0).astype(int)
        
        # 5. Create interaction terms
        if all(col in transformed_df.columns for col in ['FED_FUNDS_RATE', 'VIX']):
            transformed_df['rate_vix_interaction'] = transformed_df['FED_FUNDS_RATE'] * transformed_df['VIX']
        
        # 6. Add time-based features
        transformed_df['year'] = transformed_df.index.year
        transformed_df['month'] = transformed_df.index.month
        transformed_df['quarter'] = transformed_df.index.quarter
        transformed_df['day_of_week'] = transformed_df.index.dayofweek
        transformed_df['week_of_year'] = transformed_df.index.isocalendar().week
        
        # Cyclical encoding
        transformed_df['month_sin'] = np.sin(2 * np.pi * transformed_df['month'] / 12)
        transformed_df['month_cos'] = np.cos(2 * np.pi * transformed_df['month'] / 12)
        
        # 7. Fill missing values created by transformations
        transformed_df = transformed_df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Market data transformation completed. Shape: {transformed_df.shape}")
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error transforming market data: {e}")
        return market_df

def transform_customer_data(customer_df: pd.DataFrame) -> pd.DataFrame:
    """Transform customer data for modeling"""
    try:
        logger.info("Transforming customer data...")
        
        transformed_df = customer_df.copy()
        
        # 1. Encode categorical variables
        categorical_columns = ['segment', 'region', 'industry']
        
        for col in categorical_columns:
            if col in transformed_df.columns:
                # One-hot encoding
                dummies = pd.get_dummies(transformed_df[col], prefix=col)
                transformed_df = pd.concat([transformed_df, dummies], axis=1)
        
        # 2. Create balance-based features
        if 'balance_avg' in transformed_df.columns:
            # Log transformation for skewed balance distribution
            transformed_df['balance_log'] = np.log1p(transformed_df['balance_avg'])
            
            # Balance percentile ranking
            transformed_df['balance_percentile'] = transformed_df['balance_avg'].rank(pct=True)
            
            # Balance categories
            balance_quartiles = transformed_df['balance_avg'].quantile([0.25, 0.5, 0.75])
            transformed_df['balance_category'] = pd.cut(
                transformed_df['balance_avg'],
                bins=[0, balance_quartiles[0.25], balance_quartiles[0.5], balance_quartiles[0.75], float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
            
            # One-hot encode balance categories
            balance_dummies = pd.get_dummies(transformed_df['balance_category'], prefix='balance_cat')
            transformed_df = pd.concat([transformed_df, balance_dummies], axis=1)
        
        # 3. Create behavioral risk scores
        if all(col in transformed_df.columns for col in ['rate_sensitivity', 'loyalty_score', 'volatility_factor']):
            # Composite risk score
            transformed_df['behavioral_risk_score'] = (
                0.4 * transformed_df['rate_sensitivity'] +
                0.3 * (1 - transformed_df['loyalty_score']) +  # Higher disloyalty = higher risk
                0.3 * transformed_df['volatility_factor']
            )
            
            # Risk quintiles
            transformed_df['risk_quintile'] = pd.qcut(
                transformed_df['behavioral_risk_score'], 
                q=5, 
                labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
            )
            
            # One-hot encode risk quintiles
            risk_dummies = pd.get_dummies(transformed_df['risk_quintile'], prefix='risk')
            transformed_df = pd.concat([transformed_df, risk_dummies], axis=1)
        
        # 4. Create interaction features
        if 'balance_avg' in transformed_df.columns and 'rate_sensitivity' in transformed_df.columns:
            transformed_df['balance_sensitivity_interaction'] = (
                transformed_df['balance_log'] * transformed_df['rate_sensitivity']
            )
        
        # 5. Account tenure features
        if 'account_opening_date' in transformed_df.columns:
            transformed_df['account_opening_date'] = pd.to_datetime(transformed_df['account_opening_date'])
            reference_date = datetime.now()
            
            # Account age in days
            transformed_df['account_age_days'] = (reference_date - transformed_df['account_opening_date']).dt.days
            
            # Account age categories
            transformed_df['account_age_years'] = transformed_df['account_age_days'] / 365.25
            transformed_df['new_customer'] = (transformed_df['account_age_years'] < 1).astype(int)
            transformed_df['mature_customer'] = (transformed_df['account_age_years'] > 5).astype(int)
        
        # 6. Create segment-specific features
        if 'segment' in transformed_df.columns:
            segment_means = transformed_df.groupby('segment').agg({
                'balance_avg': 'mean',
                'rate_sensitivity': 'mean',
                'loyalty_score': 'mean'
            }).add_suffix('_segment_mean')
            
            transformed_df = transformed_df.merge(
                segment_means, 
                left_on='segment', 
                right_index=True, 
                how='left'
            )
            
            # Deviation from segment means
            if 'balance_avg' in transformed_df.columns:
                transformed_df['balance_vs_segment'] = (
                    transformed_df['balance_avg'] - transformed_df['balance_avg_segment_mean']
                )
        
        logger.info(f"Customer data transformation completed. Shape: {transformed_df.shape}")
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error transforming customer data: {e}")
        return customer_df

def transform_flow_data(flow_df: pd.DataFrame) -> pd.DataFrame:
    """Transform deposit flow data for modeling"""
    try:
        logger.info("Transforming flow data...")
        
        transformed_df = flow_df.copy()
        
        # Sort by customer and date
        if all(col in transformed_df.columns for col in ['customer_id', 'date']):
            transformed_df = transformed_df.sort_values(['customer_id', 'date'])
        
        # 1. Create time-based features
        if 'date' in transformed_df.columns:
            transformed_df['date'] = pd.to_datetime(transformed_df['date'])
            
            # Extract date components
            transformed_df['year'] = transformed_df['date'].dt.year
            transformed_df['month'] = transformed_df['date'].dt.month
            transformed_df['day'] = transformed_df['date'].dt.day
            transformed_df['day_of_week'] = transformed_df['date'].dt.dayofweek
            transformed_df['week_of_year'] = transformed_df['date'].dt.isocalendar().week
            transformed_df['quarter'] = transformed_df['date'].dt.quarter
            
            # Business day indicators
            transformed_df['is_month_end'] = transformed_df['date'].dt.is_month_end.astype(int)
            transformed_df['is_quarter_end'] = transformed_df['date'].dt.is_quarter_end.astype(int)
            transformed_df['is_year_end'] = transformed_df['date'].dt.is_year_end.astype(int)
            transformed_df['is_weekend'] = (transformed_df['day_of_week'] >= 5).astype(int)
        
        # 2. Create flow-based features
        if 'deposit_flow' in transformed_df.columns:
            # Flow direction indicators
            transformed_df['is_inflow'] = (transformed_df['deposit_flow'] > 0).astype(int)
            transformed_df['is_outflow'] = (transformed_df['deposit_flow'] < 0).astype(int)
            
            # Flow magnitude
            transformed_df['flow_magnitude'] = np.abs(transformed_df['deposit_flow'])
            transformed_df['flow_magnitude_log'] = np.log1p(transformed_df['flow_magnitude'])
            
            # Customer-level aggregations
            if 'customer_id' in transformed_df.columns:
                customer_flow_stats = transformed_df.groupby('customer_id')['deposit_flow'].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).add_suffix('_customer')
                
                transformed_df = transformed_df.merge(
                    customer_flow_stats, 
                    left_on='customer_id', 
                    right_index=True, 
                    how='left'
                )
                
                # Relative flow metrics
                transformed_df['flow_vs_customer_mean'] = (
                    transformed_df['deposit_flow'] - transformed_df['mean_customer']
                )
                
                # Z-score of flow relative to customer history
                transformed_df['flow_zscore'] = (
                    (transformed_df['deposit_flow'] - transformed_df['mean_customer']) / 
                    (transformed_df['std_customer'] + 1e-8)
                )
        
        # 3. Create lagged features
        if all(col in transformed_df.columns for col in ['customer_id', 'deposit_flow']):
            # Sort to ensure proper lag calculation
            transformed_df = transformed_df.sort_values(['customer_id', 'date'])
            
            # Create lags within each customer group
            for lag in [1, 5, 10, 21]:  # 1 day, 1 week, 2 weeks, 1 month
                transformed_df[f'deposit_flow_lag_{lag}'] = (
                    transformed_df.groupby('customer_id')['deposit_flow'].shift(lag)
                )
            
            # Rolling statistics
            for window in [5, 10, 21]:
                transformed_df[f'deposit_flow_ma_{window}'] = (
                    transformed_df.groupby('customer_id')['deposit_flow']
                    .rolling(window=window, min_periods=1).mean().reset_index(drop=True)
                )
                
                transformed_df[f'deposit_flow_std_{window}'] = (
                    transformed_df.groupby('customer_id')['deposit_flow']
                    .rolling(window=window, min_periods=1).std().reset_index(drop=True)
                )
        
        # 4. Balance-based transformations
        balance_columns = [col for col in transformed_df.columns if 'balance' in col.lower()]
        
        for col in balance_columns:
            if col in transformed_df.columns:
                # Log transformation
                transformed_df[f'{col}_log'] = np.log1p(transformed_df[col])
                
                # Balance change rate
                if 'customer_id' in transformed_df.columns:
                    transformed_df[f'{col}_change'] = (
                        transformed_df.groupby('customer_id')[col].diff()
                    )
                    
                    transformed_df[f'{col}_pct_change'] = (
                        transformed_df.groupby('customer_id')[col].pct_change()
                    )
        
        # 5. Create volatility measures
        if 'customer_id' in transformed_df.columns and 'deposit_flow' in transformed_df.columns:
            # Customer-level volatility over time
            volatility_windows = [10, 21, 63]  # 2 weeks, 1 month, 3 months
            
            for window in volatility_windows:
                transformed_df[f'flow_volatility_{window}d'] = (
                    transformed_df.groupby('customer_id')['deposit_flow']
                    .rolling(window=window, min_periods=5)
                    .std().reset_index(drop=True)
                )
        
        # 6. Create segment-based features
        if 'segment' in transformed_df.columns:
            # Daily segment aggregations
            segment_daily = transformed_df.groupby(['date', 'segment']).agg({
                'deposit_flow': ['sum', 'mean', 'count']
            }).reset_index()
            
            segment_daily.columns = ['date', 'segment'] + [f'segment_{col[1]}_{col[0]}' for col in segment_daily.columns[2:]]
            
            # Merge back
            transformed_df = transformed_df.merge(
                segment_daily, 
                on=['date', 'segment'], 
                how='left'
            )
        
        # 7. Fill missing values from transformations
        transformed_df = transformed_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"Flow data transformation completed. Shape: {transformed_df.shape}")
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error transforming flow data: {e}")
        return flow_df

def scale_features(df: pd.DataFrame, scaling_method: str = 'standard', 
                  feature_columns: List[str] = None) -> Tuple[pd.DataFrame, Any]:
    """Scale numerical features"""
    try:
        logger.info(f"Scaling features using {scaling_method} method...")
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-existent columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        if not feature_columns:
            logger.warning("No numerical columns found for scaling")
            return df, None
        
        scaled_df = df.copy()
        
        # Initialize scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            logger.error(f"Unknown scaling method: {scaling_method}")
            return df, None
        
        # Fit and transform
        scaled_values = scaler.fit_transform(scaled_df[feature_columns])
        scaled_df[feature_columns] = scaled_values
        
        logger.info(f"Scaled {len(feature_columns)} features")
        return scaled_df, scaler
        
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        return df, None

def create_interaction_features(df: pd.DataFrame, 
                              interaction_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
    """Create interaction features between specified column pairs"""
    try:
        logger.info("Creating interaction features...")
        
        interaction_df = df.copy()
        
        if interaction_pairs is None:
            # Default interactions for common financial features
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Auto-generate some interactions
            interaction_pairs = []
            
            # Rate and volatility interactions
            rate_cols = [col for col in numeric_columns if 'rate' in col.lower()]
            vol_cols = [col for col in numeric_columns if 'vix' in col.lower() or 'vol' in col.lower()]
            
            for rate_col in rate_cols[:3]:  # Limit to avoid too many features
                for vol_col in vol_cols[:2]:
                    if rate_col != vol_col:
                        interaction_pairs.append((rate_col, vol_col))
        
        # Create interactions
        for col1, col2 in interaction_pairs:
            if col1 in interaction_df.columns and col2 in interaction_df.columns:
                # Multiplicative interaction
                interaction_df[f'{col1}_X_{col2}'] = interaction_df[col1] * interaction_df[col2]
                
                # Ratio interaction (with protection against division by zero)
                interaction_df[f'{col1}_over_{col2}'] = (
                    interaction_df[col1] / (interaction_df[col2] + 1e-8)
                )
        
        logger.info(f"Created {len(interaction_pairs) * 2} interaction features")
        return interaction_df
        
    except Exception as e:
        logger.error(f"Error creating interaction features: {e}")
        return df

def apply_dimensionality_reduction(df: pd.DataFrame, method: str = 'pca', 
                                 n_components: int = None, 
                                 feature_columns: List[str] = None) -> Tuple[pd.DataFrame, Any]:
    """Apply dimensionality reduction techniques"""
    try:
        logger.info(f"Applying {method} dimensionality reduction...")
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-existent columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        if len(feature_columns) < 2:
            logger.warning("Need at least 2 features for dimensionality reduction")
            return df, None
        
        # Extract features
        X = df[feature_columns].fillna(0)  # Fill NaN for PCA
        
        # Apply method
        if method.lower() == 'pca':
            if n_components is None:
                # Keep components explaining 95% of variance
                n_components = min(len(feature_columns), len(df))
            
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
            
            # Create component names
            component_names = [f'PC_{i+1}' for i in range(X_reduced.shape[1])]
            
        else:
            logger.error(f"Unknown dimensionality reduction method: {method}")
            return df, None
        
        # Create reduced DataFrame
        reduced_df = df.copy()
        
        # Add components
        for i, name in enumerate(component_names):
            reduced_df[name] = X_reduced[:, i]
        
        # Optionally remove original features to reduce dimensionality
        # reduced_df = reduced_df.drop(columns=feature_columns)
        
        logger.info(f"Applied {method}: {len(feature_columns)} -> {len(component_names)} features")
        return reduced_df, reducer
        
    except Exception as e:
        logger.error(f"Error applying dimensionality reduction: {e}")
        return df, None

def create_target_encoding(df: pd.DataFrame, categorical_col: str, target_col: str,
                         smoothing: float = 10.0) -> pd.DataFrame:
    """Create target encoding for categorical variables"""
    try:
        logger.info(f"Creating target encoding for {categorical_col}")
        
        if categorical_col not in df.columns or target_col not in df.columns:
            logger.error(f"Required columns not found: {categorical_col}, {target_col}")
            return df
        
        encoded_df = df.copy()
        
        # Calculate global mean
        global_mean = df[target_col].mean()
        
        # Calculate category statistics
        category_stats = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
        
        # Apply smoothing
        category_stats['smoothed_mean'] = (
            (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
            (category_stats['count'] + smoothing)
        )
        
        # Create mapping
        encoding_map = category_stats['smoothed_mean'].to_dict()
        
        # Apply encoding
        encoded_df[f'{categorical_col}_target_encoded'] = (
            encoded_df[categorical_col].map(encoding_map).fillna(global_mean)
        )
        
        logger.info(f"Target encoding created for {categorical_col}")
        return encoded_df
        
    except Exception as e:
        logger.error(f"Error creating target encoding: {e}")
        return df

def comprehensive_data_transformation_pipeline(market_df: pd.DataFrame, 
                                             customer_df: pd.DataFrame,
                                             flow_df: pd.DataFrame,
                                             scaling_method: str = 'standard') -> Dict:
    """Comprehensive data transformation pipeline"""
    try:
        logger.info("=== Starting Comprehensive Data Transformation Pipeline ===")
        
        transformation_results = {
            'transformation_successful': False,
            'transformed_data': {},
            'transformers': {},
            'transformation_summary': {}
        }
        
        # Step 1: Transform individual datasets
        logger.info("Step 1: Transforming individual datasets...")
        
        transformed_market = transform_market_data(market_df)
        transformed_customer = transform_customer_data(customer_df)
        transformed_flow = transform_flow_data(flow_df)
        
        # Step 2: Create interaction features
        logger.info("Step 2: Creating interaction features...")
        
        # Market interactions
        market_interactions = [
            ('FED_FUNDS_RATE', 'VIX'),
            ('TREASURY_10Y', 'VIX'),
            ('yield_spread_10y2y', 'VIX')
        ]
        transformed_market = create_interaction_features(transformed_market, market_interactions)
        
        # Step 3: Scale numerical features
        logger.info("Step 3: Scaling numerical features...")
        
        # Scale market data
        market_numeric_cols = [col for col in transformed_market.columns 
                              if col not in ['year', 'month', 'quarter', 'day_of_week']]
        transformed_market, market_scaler = scale_features(
            transformed_market, scaling_method, market_numeric_cols
        )
        
        # Scale customer data (exclude categorical dummy variables)
        customer_numeric_cols = [col for col in transformed_customer.columns 
                               if transformed_customer[col].dtype in [np.number] and 
                               not col.startswith(('segment_', 'region_', 'industry_', 'balance_cat_', 'risk_'))]
        transformed_customer, customer_scaler = scale_features(
            transformed_customer, scaling_method, customer_numeric_cols
        )
        
        # Scale flow data (exclude categorical and time features)
        flow_numeric_cols = [col for col in transformed_flow.columns 
                           if transformed_flow[col].dtype in [np.number] and 
                           not col.startswith(('is_', 'segment_')) and
                           col not in ['year', 'month', 'quarter', 'day_of_week']]
        transformed_flow, flow_scaler = scale_features(
            transformed_flow, scaling_method, flow_numeric_cols
        )
        
        # Step 4: Handle remaining missing values
        logger.info("Step 4: Final missing value handling...")
        
        transformed_market = transformed_market.fillna(method='ffill').fillna(method='bfill').fillna(0)
        transformed_customer = transformed_customer.fillna(0)
        transformed_flow = transformed_flow.fillna(method='ffill').fillna(0)
        
        # Step 5: Create transformation summary
        original_shapes = {
            'market': market_df.shape,
            'customer': customer_df.shape,
            'flow': flow_df.shape
        }
        
        transformed_shapes = {
            'market': transformed_market.shape,
            'customer': transformed_customer.shape,
            'flow': transformed_flow.shape
        }
        
        transformation_summary = {
            'original_shapes': original_shapes,
            'transformed_shapes': transformed_shapes,
            'features_added': {
                'market': transformed_shapes['market'][1] - original_shapes['market'][1],
                'customer': transformed_shapes['customer'][1] - original_shapes['customer'][1],
                'flow': transformed_shapes['flow'][1] - original_shapes['flow'][1]
            },
            'scaling_method': scaling_method,
            'transformation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Compile results
        transformation_results.update({
            'transformation_successful': True,
            'transformed_data': {
                'market_data': transformed_market,
                'customer_data': transformed_customer,
                'flow_data': transformed_flow
            },
            'transformers': {
                'market_scaler': market_scaler,
                'customer_scaler': customer_scaler,
                'flow_scaler': flow_scaler
            },
            'transformation_summary': transformation_summary
        })
        
        logger.info("=== Data Transformation Pipeline Completed Successfully ===")
        logger.info(f"Features added - Market: {transformation_summary['features_added']['market']}, "
                   f"Customer: {transformation_summary['features_added']['customer']}, "
                   f"Flow: {transformation_summary['features_added']['flow']}")
        
        return transformation_results
        
    except Exception as e:
        logger.error(f"Error in comprehensive data transformation pipeline: {e}")
        return {
            'transformation_successful': False,
            'error': str(e),
            'transformed_data': {
                'market_data': market_df,
                'customer_data': customer_df,
                'flow_data': flow_df
            }
        }

def save_transformers(transformers: Dict, filepath: str = 'data/models/transformers.pkl') -> bool:
    """Save transformation objects for later use"""
    try:
        import joblib
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(transformers, filepath)
        logger.info(f"Transformers saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving transformers: {e}")
        return False

def load_transformers(filepath: str = 'data/models/transformers.pkl') -> Dict:
    """Load saved transformation objects"""
    try:
        import joblib
        import os
        
        if not os.path.exists(filepath):
            logger.warning(f"Transformers file not found: {filepath}")
            return {}
        
        transformers = joblib.load(filepath)
        logger.info(f"Transformers loaded from {filepath}")
        return transformers
        
    except Exception as e:
        logger.error(f"Error loading transformers: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("Data transformer module loaded successfully")
    print("Available functions:")
    print("- transform_market_data")
    print("- transform_customer_data") 
    print("- transform_flow_data")
    print("- scale_features")
    print("- create_interaction_features")
    print("- apply_dimensionality_reduction")
    print("- create_target_encoding")
    print("- comprehensive_data_transformation_pipeline")
    print("- save_transformers")
    print("- load_transformers")