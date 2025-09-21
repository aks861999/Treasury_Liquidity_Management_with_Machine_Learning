import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from src.data_collection.market_data_collector import collect_market_data
from src.data_collection.customer_data_generator import generate_complete_customer_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_all_data(n_customers: int = 5000, use_cached: bool = True) -> Dict:
    """Aggregate all data sources into a unified dataset"""
    try:
        logger.info("Starting comprehensive data aggregation...")
        
        aggregated_data = {
            'collection_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'n_customers': n_customers,
                'use_cached': use_cached
            }
        }
        
        # 1. Collect market data
        logger.info("Collecting market data...")
        market_data = collect_market_data()
        
        if market_data.empty:
            logger.error("Failed to collect market data")
            return {}
        
        aggregated_data['market_data'] = market_data
        logger.info(f"Market data collected: {market_data.shape}")
        
        # 2. Generate customer data
        logger.info("Generating customer dataset...")
        customers, balances, flows = generate_complete_customer_dataset(
            n_customers=n_customers, 
            market_data=market_data
        )
        
        if customers.empty:
            logger.error("Failed to generate customer data")
            return {}
        
        aggregated_data['customer_data'] = {
            'customers': customers,
            'balances': balances,
            'flows': flows
        }
        
        logger.info(f"Customer data generated: {len(customers)} customers")
        
        # 3. Create unified time series
        unified_timeseries = create_unified_timeseries(market_data, flows)
        aggregated_data['unified_timeseries'] = unified_timeseries
        
        # 4. Data quality summary
        data_quality = assess_data_quality(aggregated_data)
        aggregated_data['data_quality'] = data_quality
        
        # 5. Save aggregated data
        save_aggregated_data(aggregated_data)
        
        logger.info("Data aggregation completed successfully")
        return aggregated_data
        
    except Exception as e:
        logger.error(f"Error in data aggregation: {e}")
        return {}

def create_unified_timeseries(market_data: pd.DataFrame, flows_data: pd.DataFrame) -> pd.DataFrame:
    """Create unified time series combining market and flow data"""
    try:
        logger.info("Creating unified time series...")
        
        # Aggregate flows by date
        if not flows_data.empty:
            daily_flows = flows_data.groupby(['date', 'segment']).agg({
                'deposit_flow': ['sum', 'mean', 'std', 'count'],
                'inflow': 'sum',
                'outflow': 'sum'
            }).reset_index()
            
            # Flatten column names
            daily_flows.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] 
                for col in daily_flows.columns
            ]
            
            # Pivot by segment
            flow_pivot = daily_flows.pivot_table(
                index='date',
                columns='segment',
                values=['deposit_flow_sum', 'deposit_flow_mean', 'inflow_sum', 'outflow_sum'],
                fill_value=0
            )
            
            # Flatten pivot columns
            flow_pivot.columns = [f"{col[0]}_{col[1]}" for col in flow_pivot.columns]
            flow_pivot = flow_pivot.reset_index()
        else:
            flow_pivot = pd.DataFrame()
        
        # Merge with market data
        if not market_data.empty:
            market_df = market_data.reset_index()
            market_df.columns = ['date'] + list(market_df.columns[1:])
            
            if not flow_pivot.empty:
                unified_df = pd.merge(market_df, flow_pivot, on='date', how='outer')
            else:
                unified_df = market_df
        else:
            unified_df = flow_pivot
        
        # Sort by date and forward fill missing values
        unified_df = unified_df.sort_values('date').fillna(method='ffill')
        
        logger.info(f"Unified time series created: {unified_df.shape}")
        return unified_df
        
    except Exception as e:
        logger.error(f"Error creating unified time series: {e}")
        return pd.DataFrame()

def assess_data_quality(aggregated_data: Dict) -> Dict:
    """Assess quality of aggregated data"""
    try:
        quality_report = {
            'assessment_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_data_quality': {},
            'customer_data_quality': {},
            'overall_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Market data quality
        market_data = aggregated_data.get('market_data', pd.DataFrame())
        if not market_data.empty:
            market_quality = {
                'total_records': len(market_data),
                'date_range': {
                    'start': market_data.index.min().strftime('%Y-%m-%d'),
                    'end': market_data.index.max().strftime('%Y-%m-%d'),
                    'days': (market_data.index.max() - market_data.index.min()).days
                },
                'missing_values_pct': (market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))) * 100,
                'completeness_score': 100 - (market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))) * 100,
                'columns': len(market_data.columns),
                'key_series_available': {
                    'fed_funds_rate': 'FED_FUNDS_RATE' in market_data.columns,
                    'treasury_10y': 'TREASURY_10Y' in market_data.columns,
                    'vix': 'VIX' in market_data.columns
                }
            }
            
            quality_report['market_data_quality'] = market_quality
        
        # Customer data quality
        customer_data = aggregated_data.get('customer_data', {})
        if customer_data:
            customers = customer_data.get('customers', pd.DataFrame())
            flows = customer_data.get('flows', pd.DataFrame())
            
            customer_quality = {
                'total_customers': len(customers) if not customers.empty else 0,
                'total_flow_records': len(flows) if not flows.empty else 0,
                'segment_distribution': customers['segment'].value_counts().to_dict() if not customers.empty else {},
                'data_completeness': {
                    'customers_complete': (customers.isnull().sum().sum() == 0) if not customers.empty else False,
                    'flows_complete': (flows.isnull().sum().sum() == 0) if not flows.empty else False
                },
                'behavioral_metrics_available': {
                    'rate_sensitivity': 'rate_sensitivity' in customers.columns if not customers.empty else False,
                    'loyalty_score': 'loyalty_score' in customers.columns if not customers.empty else False,
                    'volatility_factor': 'volatility_factor' in customers.columns if not customers.empty else False
                }
            }
            
            quality_report['customer_data_quality'] = customer_quality
        
        # Overall quality assessment
        market_score = quality_report.get('market_data_quality', {}).get('completeness_score', 0)
        customer_score = 100 if quality_report.get('customer_data_quality', {}).get('total_customers', 0) > 100 else 0
        
        overall_score = (market_score * 0.6 + customer_score * 0.4)
        quality_report['overall_score'] = overall_score
        
        # Issues and recommendations
        issues = []
        recommendations = []
        
        if market_score < 90:
            issues.append("Market data has missing values")
            recommendations.append("Implement data interpolation for missing market data")
        
        if quality_report.get('customer_data_quality', {}).get('total_customers', 0) < 1000:
            issues.append("Limited customer sample size")
            recommendations.append("Consider increasing customer sample for better model training")
        
        if overall_score < 80:
            issues.append("Overall data quality below threshold")
            recommendations.append("Review data collection process and implement quality checks")
        
        quality_report['issues'] = issues
        quality_report['recommendations'] = recommendations
        
        logger.info(f"Data quality assessment completed - Overall score: {overall_score:.1f}")
        return quality_report
        
    except Exception as e:
        logger.error(f"Error assessing data quality: {e}")
        return {}

def save_aggregated_data(aggregated_data: Dict, base_path: str = 'data/processed/') -> bool:
    """Save aggregated data to files"""
    try:
        import os
        os.makedirs(base_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save market data
        market_data = aggregated_data.get('market_data', pd.DataFrame())
        if not market_data.empty:
            market_data.to_csv(f'{base_path}market_data_{timestamp}.csv')
            logger.info(f"Market data saved: {base_path}market_data_{timestamp}.csv")
        
        # Save customer data
        customer_data = aggregated_data.get('customer_data', {})
        if customer_data:
            customers = customer_data.get('customers', pd.DataFrame())
            balances = customer_data.get('balances', pd.DataFrame())
            flows = customer_data.get('flows', pd.DataFrame())
            
            if not customers.empty:
                customers.to_csv(f'{base_path}customers_{timestamp}.csv', index=False)
            if not balances.empty:
                balances.to_csv(f'{base_path}balances_{timestamp}.csv', index=False)
            if not flows.empty:
                flows.to_csv(f'{base_path}flows_{timestamp}.csv', index=False)
            
            logger.info(f"Customer data saved with timestamp: {timestamp}")
        
        # Save unified time series
        unified_ts = aggregated_data.get('unified_timeseries', pd.DataFrame())
        if not unified_ts.empty:
            unified_ts.to_csv(f'{base_path}unified_timeseries_{timestamp}.csv', index=False)
            logger.info(f"Unified time series saved")
        
        # Save data quality report
        data_quality = aggregated_data.get('data_quality', {})
        if data_quality:
            import json
            with open(f'{base_path}data_quality_report_{timestamp}.json', 'w') as f:
                json.dump(data_quality, f, indent=2, default=str)
            logger.info(f"Data quality report saved")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving aggregated data: {e}")
        return False

def load_latest_aggregated_data(base_path: str = 'data/processed/') -> Dict:
    """Load the most recent aggregated data"""
    try:
        import os
        import glob
        
        if not os.path.exists(base_path):
            logger.warning(f"Data path does not exist: {base_path}")
            return {}
        
        # Find latest files
        market_files = glob.glob(f'{base_path}market_data_*.csv')
        customer_files = glob.glob(f'{base_path}customers_*.csv')
        
        if not market_files or not customer_files:
            logger.warning("No aggregated data files found")
            return {}
        
        # Get latest files
        latest_market = max(market_files, key=os.path.getctime)
        latest_customers = max(customer_files, key=os.path.getctime)
        
        # Extract timestamp
        timestamp = latest_market.split('_')[-1].replace('.csv', '')
        
        aggregated_data = {
            'load_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_timestamp': timestamp
        }
        
        # Load market data
        market_data = pd.read_csv(latest_market, index_col=0, parse_dates=True)
        aggregated_data['market_data'] = market_data
        
        # Load customer data
        customers = pd.read_csv(latest_customers)
        
        # Try to load other customer files
        balances_file = f'{base_path}balances_{timestamp}.csv'
        flows_file = f'{base_path}flows_{timestamp}.csv'
        
        balances = pd.read_csv(balances_file) if os.path.exists(balances_file) else pd.DataFrame()
        flows = pd.read_csv(flows_file) if os.path.exists(flows_file) else pd.DataFrame()
        
        aggregated_data['customer_data'] = {
            'customers': customers,
            'balances': balances,
            'flows': flows
        }
        
        # Load unified time series if available
        unified_file = f'{base_path}unified_timeseries_{timestamp}.csv'
        if os.path.exists(unified_file):
            unified_ts = pd.read_csv(unified_file, parse_dates=['date'])
            aggregated_data['unified_timeseries'] = unified_ts
        
        # Load data quality report if available
        quality_file = f'{base_path}data_quality_report_{timestamp}.json'
        if os.path.exists(quality_file):
            import json
            with open(quality_file, 'r') as f:
                data_quality = json.load(f)
            aggregated_data['data_quality'] = data_quality
        
        logger.info(f"Latest aggregated data loaded from timestamp: {timestamp}")
        return aggregated_data
        
    except Exception as e:
        logger.error(f"Error loading aggregated data: {e}")
        return {}

def refresh_data_if_stale(max_age_hours: int = 24) -> Dict:
    """Refresh data if existing data is too old"""
    try:
        # Try to load existing data
        existing_data = load_latest_aggregated_data()
        
        if existing_data:
            data_timestamp_str = existing_data.get('data_timestamp', '')
            if data_timestamp_str:
                data_timestamp = datetime.strptime(data_timestamp_str, '%Y%m%d_%H%M%S')
                age_hours = (datetime.now() - data_timestamp).total_seconds() / 3600
                
                if age_hours < max_age_hours:
                    logger.info(f"Using existing data (age: {age_hours:.1f} hours)")
                    return existing_data
        
        # Data is stale or doesn't exist, refresh
        logger.info("Refreshing data due to age or absence")
        fresh_data = aggregate_all_data()
        return fresh_data
        
    except Exception as e:
        logger.error(f"Error in refresh_data_if_stale: {e}")
        return aggregate_all_data()

if __name__ == "__main__":
    # Example usage
    logger.info("Starting data aggregation...")
    
    # Aggregate fresh data
    data = aggregate_all_data(n_customers=2000)
    
    if data:
        print("Data aggregation successful!")
        print(f"Market data shape: {data['market_data'].shape}")
        
        customer_data = data.get('customer_data', {})
        if customer_data:
            print(f"Customers: {len(customer_data['customers'])}")
            print(f"Flow records: {len(customer_data['flows'])}")
        
        quality = data.get('data_quality', {})
        if quality:
            print(f"Data quality score: {quality.get('overall_score', 0):.1f}/100")
    else:
        print("Data aggregation failed!")