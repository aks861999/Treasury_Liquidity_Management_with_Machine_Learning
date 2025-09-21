import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from config import CUSTOMER_CONFIG, DATA_CONFIG, RISK_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_customer_segments(n_customers: int = 10000) -> pd.DataFrame:
    """Generate customer segments with realistic distributions"""
    try:
        np.random.seed(42)  # For reproducibility
        
        customers = []
        customer_id = 1
        
        # Define segment probabilities
        segment_probs = {
            'RETAIL_SMALL': 0.5,    # 50% of customers
            'RETAIL_MEDIUM': 0.25,  # 25% of customers
            'RETAIL_LARGE': 0.15,   # 15% of customers
            'SME': 0.08,            # 8% of customers
            'CORPORATE': 0.02       # 2% of customers
        }
        
        for segment, prob in segment_probs.items():
            n_segment = int(n_customers * prob)
            segment_config = CUSTOMER_CONFIG['SEGMENTS'][segment]
            
            for _ in range(n_segment):
                # Generate customer attributes
                customer = {
                    'customer_id': f'CUST_{customer_id:06d}',
                    'segment': segment,
                    'account_opening_date': generate_random_date(),
                    'balance_avg': generate_balance(segment_config),
                    'rate_sensitivity': generate_rate_sensitivity(segment),
                    'loyalty_score': generate_loyalty_score(segment),
                    'volatility_factor': generate_volatility_factor(segment),
                    'region': np.random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST'], p=[0.3, 0.25, 0.25, 0.2]),
                    'industry': generate_industry(segment),
                    'relationship_length': np.random.randint(1, 15)  # years
                }
                customers.append(customer)
                customer_id += 1
        
        df = pd.DataFrame(customers)
        logger.info(f"Generated {len(df)} customer profiles")
        return df
        
    except Exception as e:
        logger.error(f"Error generating customer segments: {e}")
        return pd.DataFrame()

def generate_random_date() -> datetime:
    """Generate random account opening date"""
    start_date = datetime.now() - timedelta(days=3650)  # 10 years ago
    end_date = datetime.now() - timedelta(days=30)      # At least 1 month old
    
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = np.random.randint(0, days_between)
    
    return start_date + timedelta(days=random_days)

def generate_balance(segment_config: Dict) -> float:
    """Generate realistic balance based on segment"""
    min_bal = segment_config['min_balance']
    max_bal = segment_config['max_balance']
    
    if max_bal == float('inf'):
        max_bal = 50000000  # 50M cap for corporate
    
    # Use log-normal distribution for realistic balance spread
    if min_bal == 0:
        balance = np.random.lognormal(mean=np.log(max_bal/10), sigma=1.5)
        balance = max(min_bal, min(balance, max_bal))
    else:
        log_min = np.log(min_bal)
        log_max = np.log(max_bal)
        log_balance = np.random.uniform(log_min, log_max)
        balance = np.exp(log_balance)
    
    return round(balance, 2)

def generate_rate_sensitivity(segment: str) -> float:
    """Generate rate sensitivity based on segment"""
    base_sensitivity = {
        'RETAIL_SMALL': 0.2,
        'RETAIL_MEDIUM': 0.35,
        'RETAIL_LARGE': 0.5,
        'SME': 0.7,
        'CORPORATE': 0.8
    }
    
    base = base_sensitivity.get(segment, 0.3)
    # Add noise
    sensitivity = base + np.random.normal(0, 0.1)
    return max(0.1, min(1.0, sensitivity))  # Bound between 0.1 and 1.0

def generate_loyalty_score(segment: str) -> float:
    """Generate loyalty score based on segment"""
    base_loyalty = {
        'RETAIL_SMALL': 0.6,
        'RETAIL_MEDIUM': 0.7,
        'RETAIL_LARGE': 0.8,
        'SME': 0.75,
        'CORPORATE': 0.85
    }
    
    base = base_loyalty.get(segment, 0.7)
    # Add noise
    loyalty = base + np.random.normal(0, 0.15)
    return max(0.1, min(1.0, loyalty))  # Bound between 0.1 and 1.0

def generate_volatility_factor(segment: str) -> float:
    """Generate volatility factor based on segment"""
    base_volatility = {
        'RETAIL_SMALL': 0.25,
        'RETAIL_MEDIUM': 0.2,
        'RETAIL_LARGE': 0.15,
        'SME': 0.3,
        'CORPORATE': 0.1
    }
    
    base = base_volatility.get(segment, 0.2)
    # Add noise
    volatility = base + np.random.normal(0, 0.05)
    return max(0.05, min(0.5, volatility))  # Bound between 0.05 and 0.5

def generate_industry(segment: str) -> str:
    """Generate industry based on segment"""
    if segment in ['RETAIL_SMALL', 'RETAIL_MEDIUM', 'RETAIL_LARGE']:
        return 'RETAIL'
    elif segment == 'SME':
        return np.random.choice([
            'RETAIL_TRADE', 'MANUFACTURING', 'SERVICES', 'CONSTRUCTION', 
            'AGRICULTURE', 'TECHNOLOGY', 'HEALTHCARE'
        ])
    else:  # CORPORATE
        return np.random.choice([
            'FINANCIAL_SERVICES', 'MANUFACTURING', 'ENERGY', 'TECHNOLOGY',
            'HEALTHCARE', 'REAL_ESTATE', 'UTILITIES'
        ])

def simulate_daily_balances(customer_df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    """Simulate daily account balances based on customer behavior and market conditions"""
    try:
        if market_data.empty:
            logger.warning("No market data provided, using synthetic market conditions")
            market_data = create_synthetic_market_data()
        
        logger.info("Starting daily balance simulation...")
        
        # Prepare results dataframe
        balance_records = []
        
        for _, customer in customer_df.iterrows():
            customer_balances = simulate_customer_balance_series(customer, market_data)
            balance_records.extend(customer_balances)
        
        balance_df = pd.DataFrame(balance_records)
        balance_df['date'] = pd.to_datetime(balance_df['date'])
        
        logger.info(f"Generated {len(balance_df)} daily balance records")
        return balance_df
        
    except Exception as e:
        logger.error(f"Error simulating daily balances: {e}")
        return pd.DataFrame()

def simulate_customer_balance_series(customer: pd.Series, market_data: pd.DataFrame) -> List[Dict]:
    """Simulate balance series for a single customer"""
    records = []
    
    # Start from account opening date, but not earlier than market data
    start_date = max(
        pd.to_datetime(customer['account_opening_date']),
        market_data.index.min() if not market_data.empty else pd.to_datetime(DATA_CONFIG['START_DATE'])
    )
    
    end_date = pd.to_datetime(DATA_CONFIG['END_DATE'])
    current_balance = customer['balance_avg']
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for date in date_range:
        # Get market conditions for this date
        market_conditions = get_market_conditions(date, market_data)
        
        # Calculate balance change based on customer behavior and market conditions
        balance_change = calculate_balance_change(
            customer, current_balance, market_conditions, date
        )
        
        current_balance = max(0, current_balance + balance_change)
        
        # Record daily balance
        record = {
            'customer_id': customer['customer_id'],
            'date': date,
            'balance': round(current_balance, 2),
            'balance_change': round(balance_change, 2),
            'segment': customer['segment']
        }
        records.append(record)
    
    return records

def get_market_conditions(date: pd.Timestamp, market_data: pd.DataFrame) -> Dict:
    """Get market conditions for a specific date"""
    if market_data.empty:
        return create_default_market_conditions()
    
    # Find the closest available date in market data
    available_dates = market_data.index
    closest_date = available_dates[available_dates <= date].max() if len(available_dates[available_dates <= date]) > 0 else available_dates.min()
    
    if pd.isna(closest_date):
        return create_default_market_conditions()
    
    market_row = market_data.loc[closest_date]
    
    return {
        'fed_funds_rate': market_row.get('FED_FUNDS_RATE', 2.0),
        'treasury_10y': market_row.get('TREASURY_10Y', 3.0),
        'vix': market_row.get('VIX', 20.0),
        'yield_spread': market_row.get('YIELD_SPREAD_10Y2Y', 1.0),
        'rate_change': market_row.get('FED_FUNDS_RATE_CHANGE', 0.0)
    }

def create_default_market_conditions() -> Dict:
    """Create default market conditions when data is unavailable"""
    return {
        'fed_funds_rate': 2.0,
        'treasury_10y': 3.0,
        'vix': 20.0,
        'yield_spread': 1.0,
        'rate_change': 0.0
    }

def create_synthetic_market_data() -> pd.DataFrame:
    """Create synthetic market data for testing"""
    date_range = pd.date_range(
        start=DATA_CONFIG['START_DATE'],
        end=DATA_CONFIG['END_DATE'],
        freq='D'
    )
    
    # Generate synthetic rates with trends and noise
    base_rate = 2.0
    rate_trend = np.cumsum(np.random.normal(0, 0.01, len(date_range)))
    fed_funds = np.maximum(0, base_rate + rate_trend + np.random.normal(0, 0.1, len(date_range)))
    
    treasury_10y = fed_funds + 1 + np.random.normal(0, 0.2, len(date_range))
    vix = np.maximum(10, 20 + np.random.normal(0, 5, len(date_range)))
    
    synthetic_data = pd.DataFrame({
        'FED_FUNDS_RATE': fed_funds,
        'TREASURY_10Y': treasury_10y,
        'VIX': vix,
        'YIELD_SPREAD_10Y2Y': treasury_10y - fed_funds,
        'FED_FUNDS_RATE_CHANGE': np.concatenate([[0], np.diff(fed_funds)])
    }, index=date_range)
    
    return synthetic_data

def calculate_balance_change(customer: pd.Series, current_balance: float, 
                           market_conditions: Dict, date: pd.Timestamp) -> float:
    """Calculate daily balance change based on customer behavior and market conditions"""
    
    # Base daily change (small random walk)
    base_change = np.random.normal(0, current_balance * 0.001)
    
    # Rate sensitivity impact
    rate_impact = calculate_rate_impact(customer, market_conditions, current_balance)
    
    # Seasonal effects
    seasonal_impact = calculate_seasonal_impact(date, current_balance, customer['segment'])
    
    # Volatility impact (market stress)
    volatility_impact = calculate_volatility_impact(customer, market_conditions, current_balance)
    
    # Loyalty impact (reduces outflows during stress)
    loyalty_impact = calculate_loyalty_impact(customer, market_conditions, current_balance)
    
    # Combine all impacts
    total_change = base_change + rate_impact + seasonal_impact + volatility_impact + loyalty_impact
    
    # Apply customer-specific volatility factor
    total_change *= (1 + customer['volatility_factor'] * np.random.normal(0, 0.1))
    
    return total_change

def calculate_rate_impact(customer: pd.Series, market_conditions: Dict, balance: float) -> float:
    """Calculate impact of interest rate changes on deposit flows"""
    rate_change = market_conditions.get('rate_change', 0)
    rate_sensitivity = customer['rate_sensitivity']
    
    # Positive rate changes generally attract deposits, negative repel them
    rate_impact = rate_change * rate_sensitivity * balance * 0.1
    
    # Add non-linear effects for large rate changes
    if abs(rate_change) > 0.5:  # 50 bps
        rate_impact *= 1.5
    
    return rate_impact

def calculate_seasonal_impact(date: pd.Timestamp, balance: float, segment: str) -> float:
    """Calculate seasonal impact on deposit flows"""
    month = date.month
    
    # Retail segments have stronger seasonal patterns
    seasonal_multiplier = 1.0
    if segment.startswith('RETAIL'):
        seasonal_multiplier = 1.5
    elif segment == 'SME':
        seasonal_multiplier = 1.2
    
    # Monthly seasonal patterns
    seasonal_factors = {
        1: 0.1,    # January - tax season preparation
        2: 0.05,   # February
        3: -0.1,   # March - tax payments
        4: -0.05,  # April
        5: 0.0,    # May
        6: 0.1,    # June - bonuses
        7: -0.05,  # July - vacation spending
        8: -0.05,  # August
        9: 0.05,   # September
        10: 0.0,   # October
        11: -0.1,  # November - holiday spending
        12: 0.15   # December - year-end bonuses
    }
    
    seasonal_factor = seasonal_factors.get(month, 0)
    return balance * seasonal_factor * seasonal_multiplier * 0.01

def calculate_volatility_impact(customer: pd.Series, market_conditions: Dict, balance: float) -> float:
    """Calculate impact of market volatility on deposit flows"""
    vix = market_conditions.get('vix', 20)
    volatility_factor = customer['volatility_factor']
    
    # High VIX tends to drive flight to safety (deposits increase)
    if vix > 25:  # High volatility
        volatility_impact = balance * 0.002 * (vix - 25) / 10
    elif vix < 15:  # Low volatility
        volatility_impact = -balance * 0.001 * (15 - vix) / 5
    else:
        volatility_impact = 0
    
    return volatility_impact * (1 - volatility_factor)  # More volatile customers less affected by market volatility

def calculate_loyalty_impact(customer: pd.Series, market_conditions: Dict, balance: float) -> float:
    """Calculate impact of customer loyalty on deposit stability"""
    loyalty_score = customer['loyalty_score']
    rate_change = market_conditions.get('rate_change', 0)
    
    # Loyal customers are less likely to move deposits during rate changes
    if rate_change < -0.25:  # Rate cuts
        loyalty_impact = balance * 0.001 * loyalty_score  # Loyal customers stay
    elif rate_change > 0.25:  # Rate increases
        loyalty_impact = balance * 0.0005 * loyalty_score  # Loyal customers bring more
    else:
        loyalty_impact = 0
    
    return loyalty_impact

def calculate_deposit_flows(balance_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate deposit flows (inflows/outflows) from balance changes"""
    try:
        flow_records = []
        
        for customer_id in balance_df['customer_id'].unique():
            customer_data = balance_df[balance_df['customer_id'] == customer_id].sort_values('date')
            
            for i in range(1, len(customer_data)):
                current_row = customer_data.iloc[i]
                prev_row = customer_data.iloc[i-1]
                
                flow = current_row['balance'] - prev_row['balance']
                
                flow_record = {
                    'customer_id': customer_id,
                    'date': current_row['date'],
                    'deposit_flow': flow,
                    'inflow': max(0, flow),
                    'outflow': min(0, flow),
                    'segment': current_row['segment'],
                    'balance_start': prev_row['balance'],
                    'balance_end': current_row['balance']
                }
                flow_records.append(flow_record)
        
        flows_df = pd.DataFrame(flow_records)
        logger.info(f"Calculated deposit flows for {len(flows_df)} records")
        return flows_df
        
    except Exception as e:
        logger.error(f"Error calculating deposit flows: {e}")
        return pd.DataFrame()

def aggregate_daily_flows(flows_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual customer flows to daily totals by segment"""
    try:
        daily_agg = flows_df.groupby(['date', 'segment']).agg({
            'deposit_flow': 'sum',
            'inflow': 'sum',
            'outflow': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        daily_agg.rename(columns={'customer_id': 'active_customers'}, inplace=True)
        
        # Calculate net flow rates
        daily_agg['net_flow_rate'] = daily_agg['deposit_flow'] / (
            daily_agg['balance_start'] if 'balance_start' in daily_agg.columns else 1
        )
        
        logger.info(f"Aggregated to {len(daily_agg)} daily segment records")
        return daily_agg
        
    except Exception as e:
        logger.error(f"Error aggregating daily flows: {e}")
        return pd.DataFrame()

def save_customer_data(customer_df: pd.DataFrame, balance_df: pd.DataFrame, 
                      flows_df: pd.DataFrame) -> bool:
    """Save all customer-related datasets"""
    try:
        # Save customer profiles
        customer_df.to_csv('data/raw/customer_profiles.csv', index=False)
        
        # Save daily balances
        balance_df.to_csv('data/raw/customer_daily_balances.csv', index=False)
        
        # Save deposit flows
        flows_df.to_csv('data/raw/customer_deposit_flows.csv', index=False)
        
        # Save aggregated daily flows
        daily_agg = aggregate_daily_flows(flows_df)
        daily_agg.to_csv('data/raw/daily_deposit_flows_by_segment.csv', index=False)
        
        logger.info("All customer data saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error saving customer data: {e}")
        return False

# Main execution function
def generate_complete_customer_dataset(n_customers: int = 10000, 
                                     market_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main function to generate complete customer dataset"""
    logger.info("=== Starting Customer Data Generation ===")
    
    try:
        # Generate customer segments
        customer_df = generate_customer_segments(n_customers)
        
        if customer_df.empty:
            logger.error("Failed to generate customer segments")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Simulate daily balances
        balance_df = simulate_daily_balances(customer_df, market_data)
        
        if balance_df.empty:
            logger.error("Failed to simulate daily balances")
            return customer_df, pd.DataFrame(), pd.DataFrame()
        
        # Calculate deposit flows
        flows_df = calculate_deposit_flows(balance_df)
        
        # Save all data
        save_customer_data(customer_df, balance_df, flows_df)
        
        logger.info("=== Customer Data Generation Complete ===")
        logger.info(f"Generated data for {len(customer_df)} customers")
        logger.info(f"Daily balances: {len(balance_df)} records")
        logger.info(f"Deposit flows: {len(flows_df)} records")
        
        return customer_df, balance_df, flows_df
        
    except Exception as e:
        logger.error(f"Error in customer data generation: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    # Generate customer data
    customers, balances, flows = generate_complete_customer_dataset(n_customers=5000)
    
    if not customers.empty:
        print(f"Customer segments distribution:")
        print(customers['segment'].value_counts())
        
        print(f"\nBalance statistics by segment:")
        print(customers.groupby('segment')['balance_avg'].describe())
        
        if not flows.empty:
            daily_summary = flows.groupby('date').agg({
                'deposit_flow': ['sum', 'count'],
                'inflow': 'sum',
                'outflow': 'sum'
            }).round(2)
            print(f"\nDaily flow summary (last 5 days):")
            print(daily_summary.tail())