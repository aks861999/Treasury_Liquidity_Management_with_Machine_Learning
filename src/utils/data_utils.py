import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_numeric_data(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Clean and validate numeric data"""
    try:
        cleaned_df = df.copy()
        
        if columns is None:
            columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in cleaned_df.columns:
                # Replace inf with NaN
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Remove extreme outliers (beyond 5 standard deviations)
                if cleaned_df[col].std() > 0:
                    z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                    cleaned_df.loc[z_scores > 5, col] = np.nan
        
        logger.info(f"Cleaned numeric data for {len(columns)} columns")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning numeric data: {e}")
        return df

def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """Detect outliers in a pandas Series"""
    try:
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
            
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return pd.Series([False] * len(series), index=series.index)
        
        return outliers
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        return pd.Series([False] * len(series), index=series.index)

def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
    """Handle missing values with different strategies per column"""
    try:
        filled_df = df.copy()
        
        if strategy is None:
            # Default strategies
            strategy = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    strategy[col] = 'interpolate'
                else:
                    strategy[col] = 'forward_fill'
        
        for col, method in strategy.items():
            if col not in filled_df.columns:
                continue
                
            if method == 'drop':
                filled_df = filled_df.dropna(subset=[col])
            elif method == 'mean':
                filled_df[col].fillna(filled_df[col].mean(), inplace=True)
            elif method == 'median':
                filled_df[col].fillna(filled_df[col].median(), inplace=True)
            elif method == 'mode':
                mode_val = filled_df[col].mode().iloc[0] if not filled_df[col].mode().empty else 'Unknown'
                filled_df[col].fillna(mode_val, inplace=True)
            elif method == 'forward_fill':
                filled_df[col].fillna(method='ffill', inplace=True)
            elif method == 'backward_fill':
                filled_df[col].fillna(method='bfill', inplace=True)
            elif method == 'interpolate':
                if filled_df[col].dtype in ['int64', 'float64']:
                    filled_df[col] = filled_df[col].interpolate(method='linear')
                filled_df[col].fillna(method='ffill', inplace=True)
                filled_df[col].fillna(method='bfill', inplace=True)
            elif method == 'zero':
                filled_df[col].fillna(0, inplace=True)
        
        logger.info("Missing values handled successfully")
        return filled_df
        
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        return df

def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, bool]:
    """Validate data types against expected types"""
    try:
        validation_results = {}
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                validation_results[col] = False
                continue
            
            actual_type = str(df[col].dtype)
            
            # Type mapping for validation
            type_mapping = {
                'numeric': ['int64', 'int32', 'float64', 'float32'],
                'integer': ['int64', 'int32'],
                'float': ['float64', 'float32'],
                'string': ['object', 'string'],
                'datetime': ['datetime64[ns]', 'datetime64'],
                'boolean': ['bool']
            }
            
            if expected_type in type_mapping:
                validation_results[col] = actual_type in type_mapping[expected_type]
            else:
                validation_results[col] = actual_type == expected_type
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating data types: {e}")
        return {}

def convert_data_types(df: pd.DataFrame, type_conversions: Dict[str, str]) -> pd.DataFrame:
    """Convert data types safely"""
    try:
        converted_df = df.copy()
        
        for col, target_type in type_conversions.items():
            if col not in converted_df.columns:
                continue
            
            try:
                if target_type == 'numeric':
                    converted_df[col] = pd.to_numeric(converted_df[col], errors='coerce')
                elif target_type == 'datetime':
                    converted_df[col] = pd.to_datetime(converted_df[col], errors='coerce')
                elif target_type == 'string':
                    converted_df[col] = converted_df[col].astype(str)
                elif target_type == 'category':
                    converted_df[col] = converted_df[col].astype('category')
                elif target_type == 'boolean':
                    converted_df[col] = converted_df[col].astype(bool)
                else:
                    converted_df[col] = converted_df[col].astype(target_type)
                    
                logger.info(f"Converted {col} to {target_type}")
                
            except Exception as e:
                logger.warning(f"Failed to convert {col} to {target_type}: {e}")
        
        return converted_df
        
    except Exception as e:
        logger.error(f"Error converting data types: {e}")
        return df

def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Create date-based features from a datetime column"""
    try:
        feature_df = df.copy()
        
        if date_column not in feature_df.columns:
            logger.error(f"Date column {date_column} not found")
            return df
        
        # Ensure datetime type
        feature_df[date_column] = pd.to_datetime(feature_df[date_column])
        
        # Extract date components
        feature_df[f'{date_column}_year'] = feature_df[date_column].dt.year
        feature_df[f'{date_column}_month'] = feature_df[date_column].dt.month
        feature_df[f'{date_column}_day'] = feature_df[date_column].dt.day
        feature_df[f'{date_column}_dayofweek'] = feature_df[date_column].dt.dayofweek
        feature_df[f'{date_column}_dayofyear'] = feature_df[date_column].dt.dayofyear
        feature_df[f'{date_column}_week'] = feature_df[date_column].dt.isocalendar().week
        feature_df[f'{date_column}_quarter'] = feature_df[date_column].dt.quarter
        
        # Boolean features
        feature_df[f'{date_column}_is_weekend'] = feature_df[date_column].dt.dayofweek >= 5
        feature_df[f'{date_column}_is_monthstart'] = feature_df[date_column].dt.is_month_start
        feature_df[f'{date_column}_is_monthend'] = feature_df[date_column].dt.is_month_end
        feature_df[f'{date_column}_is_quarterstart'] = feature_df[date_column].dt.is_quarter_start
        feature_df[f'{date_column}_is_quarterend'] = feature_df[date_column].dt.is_quarter_end
        feature_df[f'{date_column}_is_yearstart'] = feature_df[date_column].dt.is_year_start
        feature_df[f'{date_column}_is_yearend'] = feature_df[date_column].dt.is_year_end
        
        # Cyclical encoding
        feature_df[f'{date_column}_month_sin'] = np.sin(2 * np.pi * feature_df[f'{date_column}_month'] / 12)
        feature_df[f'{date_column}_month_cos'] = np.cos(2 * np.pi * feature_df[f'{date_column}_month'] / 12)
        feature_df[f'{date_column}_dayofweek_sin'] = np.sin(2 * np.pi * feature_df[f'{date_column}_dayofweek'] / 7)
        feature_df[f'{date_column}_dayofweek_cos'] = np.cos(2 * np.pi * feature_df[f'{date_column}_dayofweek'] / 7)
        
        logger.info(f"Created date features from {date_column}")
        return feature_df
        
    except Exception as e:
        logger.error(f"Error creating date features: {e}")
        return df

def sample_data(df: pd.DataFrame, method: str = 'random', n: int = 1000, 
               stratify_column: str = None) -> pd.DataFrame:
    """Sample data using various methods"""
    try:
        if len(df) <= n:
            return df
        
        if method == 'random':
            sampled_df = df.sample(n=n, random_state=42)
        elif method == 'systematic':
            step = len(df) // n
            indices = range(0, len(df), step)[:n]
            sampled_df = df.iloc[indices]
        elif method == 'stratified' and stratify_column and stratify_column in df.columns:
            sampled_df = df.groupby(stratify_column, group_keys=False).apply(
                lambda x: x.sample(min(len(x), n // df[stratify_column].nunique()), random_state=42)
            )
        else:
            logger.warning(f"Unknown sampling method: {method}, using random")
            sampled_df = df.sample(n=n, random_state=42)
        
        logger.info(f"Sampled {len(sampled_df)} records using {method} method")
        return sampled_df
        
    except Exception as e:
        logger.error(f"Error sampling data: {e}")
        return df

def merge_datasets(datasets: Dict[str, pd.DataFrame], join_key: str, 
                  how: str = 'inner') -> pd.DataFrame:
    """Merge multiple datasets on a common key"""
    try:
        if not datasets:
            return pd.DataFrame()
        
        dataset_names = list(datasets.keys())
        merged_df = datasets[dataset_names[0]].copy()
        
        for name in dataset_names[1:]:
            df = datasets[name]
            if join_key in df.columns and join_key in merged_df.columns:
                merged_df = merged_df.merge(df, on=join_key, how=how, suffixes=('', f'_{name}'))
                logger.info(f"Merged {name} dataset")
            else:
                logger.warning(f"Join key {join_key} not found in {name}, skipping")
        
        logger.info(f"Final merged dataset shape: {merged_df.shape}")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        return pd.DataFrame()

def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson', 
                               min_periods: int = 10) -> pd.DataFrame:
    """Calculate correlation matrix with various methods"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation")
            return pd.DataFrame()
        
        corr_matrix = numeric_df.corr(method=method, min_periods=min_periods)
        
        logger.info(f"Calculated {method} correlation matrix for {len(numeric_df.columns)} columns")
        return corr_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return pd.DataFrame()

def create_summary_statistics(df: pd.DataFrame, groupby_column: str = None) -> Dict:
    """Create comprehensive summary statistics"""
    try:
        summary = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': df.dtypes.to_dict()
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_column': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
            }
        }
        
        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary['numeric_stats'] = numeric_df.describe().to_dict()
            
            # Additional numeric metrics
            summary['additional_numeric'] = {
                'skewness': numeric_df.skew().to_dict(),
                'kurtosis': numeric_df.kurtosis().to_dict(),
                'variance': numeric_df.var().to_dict()
            }
        
        # Categorical statistics
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            summary['categorical_stats'] = {}
            for col in categorical_df.columns:
                summary['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'value_counts': df[col].value_counts().head(10).to_dict()
                }
        
        # Grouped statistics
        if groupby_column and groupby_column in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                grouped_stats = df.groupby(groupby_column)[numeric_cols].agg(['mean', 'std', 'count'])
                summary['grouped_stats'] = grouped_stats.to_dict()
        
        logger.info("Summary statistics created successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error creating summary statistics: {e}")
        return {}

def validate_data_consistency(df: pd.DataFrame, rules: Dict[str, Dict]) -> Dict[str, bool]:
    """Validate data against business rules"""
    try:
        validation_results = {}
        
        for rule_name, rule_config in rules.items():
            try:
                rule_type = rule_config.get('type')
                column = rule_config.get('column')
                
                if rule_type == 'range' and column in df.columns:
                    min_val = rule_config.get('min')
                    max_val = rule_config.get('max')
                    valid = df[column].between(min_val, max_val).all()
                    validation_results[rule_name] = valid
                    
                elif rule_type == 'not_null' and column in df.columns:
                    valid = df[column].notnull().all()
                    validation_results[rule_name] = valid
                    
                elif rule_type == 'unique' and column in df.columns:
                    valid = df[column].nunique() == len(df)
                    validation_results[rule_name] = valid
                    
                elif rule_type == 'values' and column in df.columns:
                    allowed_values = rule_config.get('allowed_values', [])
                    valid = df[column].isin(allowed_values).all()
                    validation_results[rule_name] = valid
                    
                elif rule_type == 'pattern' and column in df.columns:
                    pattern = rule_config.get('pattern')
                    valid = df[column].astype(str).str.match(pattern).all()
                    validation_results[rule_name] = valid
                    
                else:
                    validation_results[rule_name] = False
                    
            except Exception as e:
                logger.warning(f"Error validating rule {rule_name}: {e}")
                validation_results[rule_name] = False
        
        logger.info(f"Validated {len(rules)} business rules")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error in data consistency validation: {e}")
        return {}

def create_data_profile(df: pd.DataFrame) -> Dict:
    """Create comprehensive data profile"""
    try:
        profile = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        # Column profiles
        column_profiles = {}
        
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_profile.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'zeros_count': (df[col] == 0).sum(),
                    'negative_count': (df[col] < 0).sum(),
                    'outliers_count': len(detect_outliers(df[col]))
                })
            
            elif df[col].dtype == 'object':
                col_profile.update({
                    'avg_length': df[col].astype(str).str.len().mean(),
                    'max_length': df[col].astype(str).str.len().max(),
                    'min_length': df[col].astype(str).str.len().min(),
                    'most_frequent': df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else None,
                    'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                })
            
            column_profiles[col] = col_profile
        
        profile['columns'] = column_profiles
        
        # Data quality score
        profile['quality_score'] = calculate_data_quality_score(df, column_profiles)
        
        logger.info("Data profile created successfully")
        return profile
        
    except Exception as e:
        logger.error(f"Error creating data profile: {e}")
        return {}

def calculate_data_quality_score(df: pd.DataFrame, column_profiles: Dict) -> float:
    """Calculate overall data quality score (0-100)"""
    try:
        total_score = 0.0
        weights = {
            'completeness': 0.3,
            'uniqueness': 0.2,
            'consistency': 0.2,
            'validity': 0.2,
            'timeliness': 0.1
        }
        
        # Completeness (non-null percentage)
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        total_score += completeness * weights['completeness']
        
        # Uniqueness (for columns that should be unique)
        # This is simplified - in practice you'd define which columns should be unique
        uniqueness = 100  # Default high score
        total_score += uniqueness * weights['uniqueness']
        
        # Consistency (standard deviation of data types)
        consistency = 100  # Simplified
        total_score += consistency * weights['consistency']
        
        # Validity (based on data types and ranges)
        validity = 90  # Simplified
        total_score += validity * weights['validity']
        
        # Timeliness (how recent the data is)
        timeliness = 95  # Simplified
        total_score += timeliness * weights['timeliness']
        
        return round(total_score, 2)
        
    except Exception as e:
        logger.error(f"Error calculating data quality score: {e}")
        return 0.0

def split_data_by_date(df: pd.DataFrame, date_column: str, 
                      train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically for time series"""
    try:
        # Sort by date
        sorted_df = df.sort_values(date_column)
        
        n_total = len(sorted_df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = sorted_df.iloc[:n_train]
        val_df = sorted_df.iloc[n_train:n_train + n_val]
        test_df = sorted_df.iloc[n_train + n_val:]
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error splitting data by date: {e}")
        return df, pd.DataFrame(), pd.DataFrame()

def export_data(df: pd.DataFrame, filepath: str, format: str = 'csv', **kwargs) -> bool:
    """Export dataframe to various formats"""
    try:
        if format.lower() == 'csv':
            df.to_csv(filepath, **kwargs)
        elif format.lower() == 'excel':
            df.to_excel(filepath, **kwargs)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, **kwargs)
        elif format.lower() == 'json':
            df.to_json(filepath, **kwargs)
        elif format.lower() == 'pickle':
            df.to_pickle(filepath)
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        logger.info(f"Data exported to {filepath} in {format} format")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return False

def load_data(filepath: str, format: str = None, **kwargs) -> pd.DataFrame:
    """Load data from various formats"""
    try:
        if format is None:
            format = filepath.split('.')[-1]
        
        if format.lower() == 'csv':
            df = pd.read_csv(filepath, **kwargs)
        elif format.lower() in ['xlsx', 'xls', 'excel']:
            df = pd.read_excel(filepath, **kwargs)
        elif format.lower() == 'parquet':
            df = pd.read_parquet(filepath, **kwargs)
        elif format.lower() == 'json':
            df = pd.read_json(filepath, **kwargs)
        elif format.lower() == 'pickle':
            df = pd.read_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return pd.DataFrame()
        
        logger.info(f"Data loaded from {filepath}: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    logger.info("Data utils module loaded successfully")
    print("Available functions:")
    print("- clean_numeric_data")
    print("- detect_outliers")
    print("- handle_missing_values")
    print("- validate_data_types")
    print("- convert_data_types")
    print("- create_date_features")
    print("- sample_data")
    print("- merge_datasets")
    print("- calculate_correlation_matrix")
    print("- create_summary_statistics")
    print("- validate_data_consistency")
    print("- create_data_profile")
    print("- split_data_by_date")
    print("- export_data")
    print("- load_data")