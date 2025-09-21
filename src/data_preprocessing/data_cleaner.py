import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from src.utils.data_utils import detect_outliers, handle_missing_values, validate_data_consistency

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_market_data(market_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate market data"""
    try:
        logger.info("Starting market data cleaning...")
        
        cleaned_df = market_df.copy()
        
        # 1. Handle missing values specific to market data
        market_missing_strategy = {
            'FED_FUNDS_RATE': 'forward_fill',
            'TREASURY_10Y': 'interpolate',
            'TREASURY_2Y': 'interpolate',
            'TREASURY_3M': 'interpolate',
            'VIX': 'forward_fill',
            'EURUSD': 'forward_fill',
            'GBPUSD': 'forward_fill'
        }
        
        cleaned_df = handle_missing_values(cleaned_df, market_missing_strategy)
        
        # 2. Validate interest rate ranges
        rate_columns = [col for col in cleaned_df.columns if 'RATE' in col or 'TREASURY' in col]
        for col in rate_columns:
            if col in cleaned_df.columns:
                # Interest rates should be between -2% and 20%
                mask = (cleaned_df[col] < -2.0) | (cleaned_df[col] > 20.0)
                if mask.any():
                    logger.warning(f"Found {mask.sum()} invalid values in {col}")
                    cleaned_df.loc[mask, col] = np.nan
        
        # 3. Clean VIX data
        if 'VIX' in cleaned_df.columns:
            # VIX should be between 8 and 100
            vix_mask = (cleaned_df['VIX'] < 8) | (cleaned_df['VIX'] > 100)
            if vix_mask.any():
                logger.warning(f"Found {vix_mask.sum()} invalid VIX values")
                cleaned_df.loc[vix_mask, 'VIX'] = np.nan
        
        # 4. Clean FX rates
        fx_columns = [col for col in cleaned_df.columns if any(fx in col for fx in ['USD', 'EUR', 'GBP', 'JPY'])]
        for col in fx_columns:
            if col in cleaned_df.columns:
                # FX rates should be positive and within reasonable ranges
                fx_mask = (cleaned_df[col] <= 0) | (cleaned_df[col] > 10)
                if fx_mask.any():
                    logger.warning(f"Found {fx_mask.sum()} invalid FX values in {col}")
                    cleaned_df.loc[fx_mask, col] = np.nan
        
        # 5. Remove duplicate dates
        if cleaned_df.index.duplicated().any():
            logger.warning(f"Found {cleaned_df.index.duplicated().sum()} duplicate dates")
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
        
        # 6. Sort by date
        cleaned_df = cleaned_df.sort_index()
        
        # 7. Final interpolation for remaining missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
        
        # Forward and backward fill for any remaining gaps
        cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Market data cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning market data: {e}")
        return market_df

def clean_customer_data(customer_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate customer data"""
    try:
        logger.info("Starting customer data cleaning...")
        
        cleaned_df = customer_df.copy()
        
        # 1. Remove duplicate customers
        if 'customer_id' in cleaned_df.columns:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=['customer_id'], keep='first')
            removed_count = initial_count - len(cleaned_df)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} duplicate customers")
        
        # 2. Validate balance ranges
        if 'balance_avg' in cleaned_df.columns:
            # Balance should be non-negative and reasonable
            balance_mask = (cleaned_df['balance_avg'] < 0) | (cleaned_df['balance_avg'] > 100_000_000)
            if balance_mask.any():
                logger.warning(f"Found {balance_mask.sum()} customers with invalid balances")
                cleaned_df = cleaned_df[~balance_mask]
        
        # 3. Validate behavioral scores
        behavioral_columns = ['rate_sensitivity', 'loyalty_score', 'volatility_factor']
        for col in behavioral_columns:
            if col in cleaned_df.columns:
                # These should be between 0 and 1
                score_mask = (cleaned_df[col] < 0) | (cleaned_df[col] > 1)
                if score_mask.any():
                    logger.warning(f"Found {score_mask.sum()} invalid {col} values")
                    # Clip to valid range
                    cleaned_df[col] = cleaned_df[col].clip(0, 1)
        
        # 4. Validate segments
        if 'segment' in cleaned_df.columns:
            valid_segments = ['RETAIL_SMALL', 'RETAIL_MEDIUM', 'RETAIL_LARGE', 'SME', 'CORPORATE']
            invalid_segments = ~cleaned_df['segment'].isin(valid_segments)
            if invalid_segments.any():
                logger.warning(f"Found {invalid_segments.sum()} customers with invalid segments")
                cleaned_df = cleaned_df[~invalid_segments]
        
        # 5. Validate dates
        if 'account_opening_date' in cleaned_df.columns:
            cleaned_df['account_opening_date'] = pd.to_datetime(cleaned_df['account_opening_date'])
            
            # Remove future dates or dates before 1990
            min_date = pd.to_datetime('1990-01-01')
            max_date = pd.to_datetime('today')
            
            date_mask = (cleaned_df['account_opening_date'] < min_date) | (cleaned_df['account_opening_date'] > max_date)
            if date_mask.any():
                logger.warning(f"Found {date_mask.sum()} customers with invalid opening dates")
                cleaned_df = cleaned_df[~date_mask]
        
        # 6. Handle missing values
        customer_missing_strategy = {
            'balance_avg': 'median',
            'rate_sensitivity': 'mean',
            'loyalty_score': 'mean',
            'volatility_factor': 'mean',
            'region': 'mode',
            'industry': 'mode'
        }
        
        cleaned_df = handle_missing_values(cleaned_df, customer_missing_strategy)
        
        logger.info(f"Customer data cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning customer data: {e}")
        return customer_df

def clean_flow_data(flow_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate deposit flow data"""
    try:
        logger.info("Starting flow data cleaning...")
        
        cleaned_df = flow_df.copy()
        
        # 1. Remove duplicate records
        if all(col in cleaned_df.columns for col in ['customer_id', 'date']):
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=['customer_id', 'date'], keep='last')
            removed_count = initial_count - len(cleaned_df)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} duplicate flow records")
        
        # 2. Validate dates
        if 'date' in cleaned_df.columns:
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
            
            # Remove future dates
            future_mask = cleaned_df['date'] > pd.to_datetime('today')
            if future_mask.any():
                logger.warning(f"Removed {future_mask.sum()} future-dated records")
                cleaned_df = cleaned_df[~future_mask]
        
        # 3. Detect and handle extreme outliers in flows
        flow_columns = [col for col in cleaned_df.columns if 'flow' in col.lower()]
        
        for col in flow_columns:
            if col in cleaned_df.columns:
                # Detect outliers using IQR method
                outliers = detect_outliers(cleaned_df[col], method='iqr', threshold=3.0)
                
                if outliers.any():
                    logger.warning(f"Found {outliers.sum()} outliers in {col}")
                    
                    # Cap extreme outliers rather than removing them
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
        
        # 4. Validate balance consistency
        if all(col in cleaned_df.columns for col in ['balance_start', 'balance_end', 'deposit_flow']):
            # Check if balance_end = balance_start + deposit_flow
            calculated_balance = cleaned_df['balance_start'] + cleaned_df['deposit_flow']
            balance_diff = abs(cleaned_df['balance_end'] - calculated_balance)
            
            # Allow for small rounding errors
            inconsistent_mask = balance_diff > 0.01
            if inconsistent_mask.any():
                logger.warning(f"Found {inconsistent_mask.sum()} records with inconsistent balances")
                # Fix by recalculating balance_end
                cleaned_df.loc[inconsistent_mask, 'balance_end'] = calculated_balance
        
        # 5. Ensure non-negative balances
        balance_columns = [col for col in cleaned_df.columns if 'balance' in col.lower()]
        for col in balance_columns:
            if col in cleaned_df.columns:
                negative_mask = cleaned_df[col] < 0
                if negative_mask.any():
                    logger.warning(f"Found {negative_mask.sum()} negative balances in {col}")
                    cleaned_df.loc[negative_mask, col] = 0
        
        # 6. Sort by customer and date
        if all(col in cleaned_df.columns for col in ['customer_id', 'date']):
            cleaned_df = cleaned_df.sort_values(['customer_id', 'date'])
        
        # 7. Handle missing values
        flow_missing_strategy = {
            'deposit_flow': 'zero',
            'inflow': 'zero',
            'outflow': 'zero',
            'balance': 'forward_fill'
        }
        
        cleaned_df = handle_missing_values(cleaned_df, flow_missing_strategy)
        
        logger.info(f"Flow data cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning flow data: {e}")
        return flow_df

def validate_data_integrity(market_df: pd.DataFrame, customer_df: pd.DataFrame, 
                          flow_df: pd.DataFrame) -> Dict[str, bool]:
    """Validate integrity across all datasets"""
    try:
        logger.info("Validating data integrity across datasets...")
        
        validation_results = {
            'market_data_valid': True,
            'customer_data_valid': True,
            'flow_data_valid': True,
            'cross_dataset_valid': True,
            'issues': []
        }
        
        # 1. Check date ranges consistency
        if not market_df.empty:
            market_date_range = (market_df.index.min(), market_df.index.max())
            
            if not flow_df.empty and 'date' in flow_df.columns:
                flow_date_range = (flow_df['date'].min(), flow_df['date'].max())
                
                # Flow dates should be within market data range
                if (flow_date_range[0] < market_date_range[0] or 
                    flow_date_range[1] > market_date_range[1]):
                    validation_results['cross_dataset_valid'] = False
                    validation_results['issues'].append("Flow data dates extend beyond market data range")
        
        # 2. Check customer-flow consistency
        if not customer_df.empty and not flow_df.empty:
            if 'customer_id' in customer_df.columns and 'customer_id' in flow_df.columns:
                customer_ids = set(customer_df['customer_id'])
                flow_customer_ids = set(flow_df['customer_id'])
                
                orphaned_flows = flow_customer_ids - customer_ids
                if orphaned_flows:
                    validation_results['cross_dataset_valid'] = False
                    validation_results['issues'].append(f"Found {len(orphaned_flows)} customer IDs in flow data without customer records")
        
        # 3. Check for reasonable data volumes
        min_market_records = 30  # At least 30 days
        min_customers = 100
        min_flow_records = 1000
        
        if len(market_df) < min_market_records:
            validation_results['market_data_valid'] = False
            validation_results['issues'].append(f"Insufficient market data records: {len(market_df)} < {min_market_records}")
        
        if len(customer_df) < min_customers:
            validation_results['customer_data_valid'] = False
            validation_results['issues'].append(f"Insufficient customer records: {len(customer_df)} < {min_customers}")
        
        if len(flow_df) < min_flow_records:
            validation_results['flow_data_valid'] = False
            validation_results['issues'].append(f"Insufficient flow records: {len(flow_df)} < {min_flow_records}")
        
        # 4. Check for critical columns
        required_market_cols = ['FED_FUNDS_RATE', 'VIX']
        required_customer_cols = ['customer_id', 'segment', 'balance_avg']
        required_flow_cols = ['customer_id', 'date', 'deposit_flow']
        
        missing_market_cols = [col for col in required_market_cols if col not in market_df.columns]
        missing_customer_cols = [col for col in required_customer_cols if col not in customer_df.columns]
        missing_flow_cols = [col for col in required_flow_cols if col not in flow_df.columns]
        
        if missing_market_cols:
            validation_results['market_data_valid'] = False
            validation_results['issues'].append(f"Missing critical market columns: {missing_market_cols}")
        
        if missing_customer_cols:
            validation_results['customer_data_valid'] = False
            validation_results['issues'].append(f"Missing critical customer columns: {missing_customer_cols}")
        
        if missing_flow_cols:
            validation_results['flow_data_valid'] = False
            validation_results['issues'].append(f"Missing critical flow columns: {missing_flow_cols}")
        
        # Overall validation
        overall_valid = all([
            validation_results['market_data_valid'],
            validation_results['customer_data_valid'],
            validation_results['flow_data_valid'],
            validation_results['cross_dataset_valid']
        ])
        
        validation_results['overall_valid'] = overall_valid
        
        if overall_valid:
            logger.info("Data integrity validation passed")
        else:
            logger.warning(f"Data integrity validation failed: {len(validation_results['issues'])} issues found")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating data integrity: {e}")
        return {'overall_valid': False, 'issues': [str(e)]}

def remove_anomalous_patterns(df: pd.DataFrame, pattern_columns: List[str]) -> pd.DataFrame:
    """Remove records with anomalous patterns"""
    try:
        logger.info("Detecting and removing anomalous patterns...")
        
        cleaned_df = df.copy()
        
        for col in pattern_columns:
            if col not in cleaned_df.columns:
                continue
            
            # Detect constant values (likely data errors)
            if cleaned_df[col].nunique() == 1:
                logger.warning(f"Column {col} has constant values - potential data issue")
                continue
            
            # Detect impossible sequences (e.g., too many consecutive identical values)
            if cleaned_df[col].dtype in [np.number]:
                # Find runs of identical values
                consecutive_mask = cleaned_df[col].diff() == 0
                max_consecutive = 0
                current_consecutive = 0
                
                for is_consecutive in consecutive_mask:
                    if is_consecutive:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 0
                
                # If more than 10 consecutive identical values, flag as suspicious
                if max_consecutive > 10:
                    logger.warning(f"Column {col} has {max_consecutive} consecutive identical values")
        
        # Remove records with too many missing values
        missing_threshold = 0.5  # Remove records with >50% missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            missing_ratio = cleaned_df[numeric_columns].isnull().sum(axis=1) / len(numeric_columns)
            anomalous_mask = missing_ratio > missing_threshold
            
            if anomalous_mask.any():
                logger.warning(f"Removing {anomalous_mask.sum()} records with excessive missing values")
                cleaned_df = cleaned_df[~anomalous_mask]
        
        logger.info(f"Anomalous pattern removal completed. Final shape: {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error removing anomalous patterns: {e}")
        return df

def apply_business_rules_cleaning(customer_df: pd.DataFrame, flow_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply business-specific cleaning rules"""
    try:
        logger.info("Applying business rules cleaning...")
        
        cleaned_customer_df = customer_df.copy()
        cleaned_flow_df = flow_df.copy()
        
        # Business Rule 1: Customers must have minimum account history
        if 'account_opening_date' in cleaned_customer_df.columns:
            min_history_days = 30
            cutoff_date = pd.to_datetime('today') - timedelta(days=min_history_days)
            
            new_account_mask = cleaned_customer_df['account_opening_date'] > cutoff_date
            if new_account_mask.any():
                logger.info(f"Flagging {new_account_mask.sum()} customers with insufficient history")
                # Don't remove, but flag for special handling
                cleaned_customer_df['insufficient_history'] = new_account_mask
        
        # Business Rule 2: Corporate customers must have minimum balance
        if 'segment' in cleaned_customer_df.columns and 'balance_avg' in cleaned_customer_df.columns:
            corporate_mask = cleaned_customer_df['segment'] == 'CORPORATE'
            min_corporate_balance = 1_000_000  # $1M minimum
            
            low_balance_corporate = corporate_mask & (cleaned_customer_df['balance_avg'] < min_corporate_balance)
            if low_balance_corporate.any():
                logger.warning(f"Found {low_balance_corporate.sum()} corporate customers below minimum balance")
                # Reclassify as SME
                cleaned_customer_df.loc[low_balance_corporate, 'segment'] = 'SME'
        
        # Business Rule 3: Remove customers with impossible flow patterns
        if 'customer_id' in cleaned_flow_df.columns:
            customer_flow_stats = cleaned_flow_df.groupby('customer_id')['deposit_flow'].agg(['std', 'mean', 'count'])
            
            # Flag customers with extremely high volatility relative to balance
            if 'balance_avg' in cleaned_customer_df.columns:
                customer_balance = cleaned_customer_df.set_index('customer_id')['balance_avg']
                
                # Calculate volatility relative to balance
                relative_volatility = customer_flow_stats['std'] / customer_balance
                extreme_volatility_mask = relative_volatility > 0.5  # 50% of balance as std
                
                if extreme_volatility_mask.any():
                    extreme_customers = extreme_volatility_mask[extreme_volatility_mask].index
                    logger.warning(f"Flagging {len(extreme_customers)} customers with extreme volatility")
                    
                    # Remove flow records for these customers
                    cleaned_flow_df = cleaned_flow_df[~cleaned_flow_df['customer_id'].isin(extreme_customers)]
        
        # Business Rule 4: Validate segment consistency with balance
        if all(col in cleaned_customer_df.columns for col in ['segment', 'balance_avg']):
            segment_balance_rules = {
                'RETAIL_SMALL': (0, 10_000),
                'RETAIL_MEDIUM': (10_000, 100_000),
                'RETAIL_LARGE': (100_000, 1_000_000),
                'SME': (50_000, 5_000_000),
                'CORPORATE': (1_000_000, float('inf'))
            }
            
            for segment, (min_bal, max_bal) in segment_balance_rules.items():
                segment_mask = cleaned_customer_df['segment'] == segment
                balance_mismatch = segment_mask & (
                    (cleaned_customer_df['balance_avg'] < min_bal) | 
                    (cleaned_customer_df['balance_avg'] > max_bal)
                )
                
                if balance_mismatch.any():
                    logger.info(f"Correcting {balance_mismatch.sum()} segment misclassifications for {segment}")
                    
                    # Reassign to correct segment based on balance
                    for idx in cleaned_customer_df[balance_mismatch].index:
                        balance = cleaned_customer_df.loc[idx, 'balance_avg']
                        
                        for correct_segment, (seg_min, seg_max) in segment_balance_rules.items():
                            if seg_min <= balance < seg_max:
                                cleaned_customer_df.loc[idx, 'segment'] = correct_segment
                                break
        
        logger.info("Business rules cleaning completed")
        return cleaned_customer_df, cleaned_flow_df
        
    except Exception as e:
        logger.error(f"Error applying business rules cleaning: {e}")
        return customer_df, flow_df

def create_data_quality_report(original_data: Dict[str, pd.DataFrame], 
                             cleaned_data: Dict[str, pd.DataFrame]) -> Dict:
    """Create comprehensive data quality report"""
    try:
        logger.info("Creating data quality report...")
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {},
            'detailed_changes': {},
            'data_quality_scores': {}
        }
        
        for dataset_name in ['market_data', 'customer_data', 'flow_data']:
            if dataset_name in original_data and dataset_name in cleaned_data:
                original_df = original_data[dataset_name]
                cleaned_df = cleaned_data[dataset_name]
                
                # Calculate changes
                original_shape = original_df.shape
                cleaned_shape = cleaned_df.shape
                
                records_removed = original_shape[0] - cleaned_shape[0]
                columns_removed = original_shape[1] - cleaned_shape[1]
                
                # Missing values comparison
                original_missing = original_df.isnull().sum().sum()
                cleaned_missing = cleaned_df.isnull().sum().sum()
                
                # Data quality score (0-100)
                completeness_score = (1 - cleaned_missing / (cleaned_df.size + 1)) * 100
                consistency_score = 95 if records_removed < original_shape[0] * 0.05 else 80
                validity_score = 90  # Simplified
                
                overall_score = (completeness_score * 0.4 + consistency_score * 0.3 + validity_score * 0.3)
                
                report['summary'][dataset_name] = {
                    'original_records': original_shape[0],
                    'cleaned_records': cleaned_shape[0],
                    'records_removed': records_removed,
                    'removal_percentage': (records_removed / original_shape[0]) * 100 if original_shape[0] > 0 else 0,
                    'original_missing_values': original_missing,
                    'cleaned_missing_values': cleaned_missing,
                    'missing_values_fixed': original_missing - cleaned_missing
                }
                
                report['data_quality_scores'][dataset_name] = {
                    'completeness': completeness_score,
                    'consistency': consistency_score,
                    'validity': validity_score,
                    'overall_score': overall_score
                }
        
        # Overall summary
        total_original_records = sum(report['summary'][ds]['original_records'] for ds in report['summary'])
        total_cleaned_records = sum(report['summary'][ds]['cleaned_records'] for ds in report['summary'])
        total_records_removed = total_original_records - total_cleaned_records
        
        report['overall_summary'] = {
            'total_original_records': total_original_records,
            'total_cleaned_records': total_cleaned_records,
            'total_records_removed': total_records_removed,
            'overall_removal_percentage': (total_records_removed / total_original_records) * 100 if total_original_records > 0 else 0,
            'cleaning_efficiency': 'High' if total_records_removed / total_original_records < 0.05 else 'Medium'
        }
        
        logger.info("Data quality report created successfully")
        return report
        
    except Exception as e:
        logger.error(f"Error creating data quality report: {e}")
        return {}

def comprehensive_data_cleaning_pipeline(market_df: pd.DataFrame, customer_df: pd.DataFrame, 
                                       flow_df: pd.DataFrame) -> Dict:
    """Complete data cleaning pipeline"""
    try:
        logger.info("=== Starting Comprehensive Data Cleaning Pipeline ===")
        
        # Store original data for reporting
        original_data = {
            'market_data': market_df.copy(),
            'customer_data': customer_df.copy(),
            'flow_data': flow_df.copy()
        }
        
        # Step 1: Individual dataset cleaning
        logger.info("Step 1: Cleaning individual datasets...")
        cleaned_market_df = clean_market_data(market_df)
        cleaned_customer_df = clean_customer_data(customer_df)
        cleaned_flow_df = clean_flow_data(flow_df)
        
        # Step 2: Apply business rules
        logger.info("Step 2: Applying business rules...")
        cleaned_customer_df, cleaned_flow_df = apply_business_rules_cleaning(cleaned_customer_df, cleaned_flow_df)
        
        # Step 3: Remove anomalous patterns
        logger.info("Step 3: Removing anomalous patterns...")
        market_pattern_cols = ['FED_FUNDS_RATE', 'VIX', 'TREASURY_10Y']
        customer_pattern_cols = ['balance_avg', 'rate_sensitivity', 'loyalty_score']
        flow_pattern_cols = ['deposit_flow', 'balance']
        
        cleaned_market_df = remove_anomalous_patterns(cleaned_market_df, market_pattern_cols)
        cleaned_customer_df = remove_anomalous_patterns(cleaned_customer_df, customer_pattern_cols)
        cleaned_flow_df = remove_anomalous_patterns(cleaned_flow_df, flow_pattern_cols)
        
        # Step 4: Validate data integrity
        logger.info("Step 4: Validating data integrity...")
        integrity_results = validate_data_integrity(cleaned_market_df, cleaned_customer_df, cleaned_flow_df)
        
        # Step 5: Create quality report
        logger.info("Step 5: Creating data quality report...")
        cleaned_data = {
            'market_data': cleaned_market_df,
            'customer_data': cleaned_customer_df,
            'flow_data': cleaned_flow_df
        }
        
        quality_report = create_data_quality_report(original_data, cleaned_data)
        
        # Compile results
        results = {
            'cleaned_data': cleaned_data,
            'data_quality_report': quality_report,
            'integrity_validation': integrity_results,
            'cleaning_successful': integrity_results.get('overall_valid', False)
        }
        
        logger.info("=== Data Cleaning Pipeline Completed ===")
        logger.info(f"Cleaning successful: {results['cleaning_successful']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive data cleaning pipeline: {e}")
        return {
            'cleaned_data': {
                'market_data': market_df,
                'customer_data': customer_df,
                'flow_data': flow_df
            },
            'cleaning_successful': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Example usage
    logger.info("Data cleaner module loaded successfully")
    print("Available functions:")
    print("- clean_market_data")
    print("- clean_customer_data")
    print("- clean_flow_data")
    print("- validate_data_integrity")
    print("- apply_business_rules_cleaning")
    print("- comprehensive_data_cleaning_pipeline")