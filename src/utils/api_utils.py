import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIRateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 100, time_window: int = 3600):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]) + 1
            logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
            self.calls = []
        
        self.calls.append(now)

def make_api_request(url: str, params: Dict = None, headers: Dict = None, 
                    timeout: int = 30, max_retries: int = 3) -> Optional[Dict]:
    """Make API request with error handling and retries"""
    try:
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"API request failed with status {response.status_code}")
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                        
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return None
        
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None

def fetch_fred_data(series_id: str, api_key: str, start_date: str = None, 
                   end_date: str = None) -> pd.Series:
    """Fetch data from FRED API"""
    try:
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'sort_order': 'asc'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        response_data = make_api_request(base_url, params)
        
        if response_data and 'observations' in response_data:
            observations = response_data['observations']
            
            dates = []
            values = []
            
            for obs in observations:
                if obs['value'] != '.':  # FRED uses '.' for missing values
                    dates.append(pd.to_datetime(obs['date']))
                    values.append(float(obs['value']))
            
            series = pd.Series(values, index=dates, name=series_id)
            logger.info(f"Fetched FRED data for {series_id}: {len(series)} observations")
            return series
        
        logger.warning(f"No data returned for FRED series {series_id}")
        return pd.Series()
        
    except Exception as e:
        logger.error(f"Error fetching FRED data for {series_id}: {e}")
        return pd.Series()

def fetch_treasury_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch Treasury yield data"""
    try:
        base_url = "https://api.fiscaldata.treasury.gov/services/api/v1/accounting/od/avg_interest_rates"
        
        params = {
            'format': 'json',
            'sort': '-record_date'
        }
        
        if start_date:
            params['filter'] = f'record_date:gte:{start_date}'
        if end_date and start_date:
            params['filter'] += f',record_date:lte:{end_date}'
        elif end_date:
            params['filter'] = f'record_date:lte:{end_date}'
        
        response_data = make_api_request(base_url, params)
        
        if response_data and 'data' in response_data:
            df = pd.DataFrame(response_data['data'])
            
            if not df.empty:
                df['record_date'] = pd.to_datetime(df['record_date'])
                df = df.set_index('record_date')
                logger.info(f"Fetched Treasury data: {len(df)} records")
                return df
        
        logger.warning("No Treasury data returned")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching Treasury data: {e}")
        return pd.DataFrame()

def fetch_ecb_data(series_key: str, start_date: str = None, end_date: str = None) -> pd.Series:
    """Fetch data from ECB Statistical Data Warehouse"""
    try:
        base_url = f"https://sdw-wsrest.ecb.europa.eu/service/data/{series_key}"
        
        headers = {'Accept': 'application/json'}
        params = {}
        
        if start_date and end_date:
            params['startPeriod'] = start_date
            params['endPeriod'] = end_date
        
        response_data = make_api_request(base_url, params, headers)
        
        if response_data and 'dataSets' in response_data:
            # ECB API response parsing (simplified)
            dataset = response_data['dataSets'][0]
            
            if 'observations' in dataset:
                observations = dataset['observations']
                structure = response_data.get('structure', {})
                dimensions = structure.get('dimensions', {}).get('observation', [])
                
                # Find time dimension
                time_dim = None
                for dim in dimensions:
                    if dim.get('id') == 'TIME_PERIOD':
                        time_dim = dim
                        break
                
                if time_dim and 'values' in time_dim:
                    time_values = [v['id'] for v in time_dim['values']]
                    
                    dates = []
                    values = []
                    
                    for i, obs_data in observations.items():
                        if obs_data and len(obs_data) > 0:
                            time_index = int(i.split(':')[0]) if ':' in i else int(i)
                            if time_index < len(time_values):
                                date_str = time_values[time_index]
                                dates.append(pd.to_datetime(date_str))
                                values.append(float(obs_data[0]))
                    
                    if dates and values:
                        series = pd.Series(values, index=dates, name=series_key)
                        logger.info(f"Fetched ECB data for {series_key}: {len(series)} observations")
                        return series
        
        logger.warning(f"No ECB data returned for {series_key}")
        return pd.Series()
        
    except Exception as e:
        logger.error(f"Error fetching ECB data for {series_key}: {e}")
        return pd.Series()

def validate_api_response(response_data: Dict, required_fields: List[str]) -> bool:
    """Validate API response has required fields"""
    try:
        if not isinstance(response_data, dict):
            return False
        
        for field in required_fields:
            if field not in response_data:
                logger.warning(f"Required field '{field}' missing from API response")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating API response: {e}")
        return False

def cache_api_response(cache_key: str, response_data: Any, cache_dir: str = "cache/") -> bool:
    """Cache API response to disk"""
    try:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        cache_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': response_data
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_entry, f, default=str, indent=2)
        
        logger.debug(f"Cached API response to {cache_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error caching API response: {e}")
        return False

def load_cached_response(cache_key: str, max_age_hours: int = 24, 
                        cache_dir: str = "cache/") -> Optional[Any]:
    """Load cached API response if not too old"""
    try:
        import os
        
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        with open(cache_file, 'r') as f:
            cache_entry = json.load(f)
        
        # Check age
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if age_hours <= max_age_hours:
            logger.debug(f"Using cached response (age: {age_hours:.1f}h)")
            return cache_entry['data']
        else:
            logger.debug(f"Cache expired (age: {age_hours:.1f}h)")
            return None
        
    except Exception as e:
        logger.error(f"Error loading cached response: {e}")
        return None

def batch_api_requests(requests_list: List[Dict], rate_limiter: APIRateLimiter = None, 
                      delay_between_requests: float = 1.0) -> List[Optional[Dict]]:
    """Execute multiple API requests with rate limiting"""
    try:
        if rate_limiter is None:
            rate_limiter = APIRateLimiter()
        
        results = []
        
        for i, request_config in enumerate(requests_list):
            logger.info(f"Processing request {i+1}/{len(requests_list)}")
            
            # Rate limiting
            rate_limiter.wait_if_needed()
            
            # Make request
            url = request_config.get('url')
            params = request_config.get('params', {})
            headers = request_config.get('headers', {})
            timeout = request_config.get('timeout', 30)
            
            result = make_api_request(url, params, headers, timeout)
            results.append(result)
            
            # Delay between requests
            if i < len(requests_list) - 1 and delay_between_requests > 0:
                time.sleep(delay_between_requests)
        
        logger.info(f"Completed {len(results)} API requests")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch API requests: {e}")
        return []

def fetch_multiple_fred_series(series_dict: Dict[str, str], api_key: str,
                              start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch multiple FRED series and combine into DataFrame"""
    try:
        rate_limiter = APIRateLimiter(max_calls=120, time_window=60)  # FRED allows 120/minute
        
        all_series = {}
        
        for series_name, series_id in series_dict.items():
            rate_limiter.wait_if_needed()
            
            series_data = fetch_fred_data(series_id, api_key, start_date, end_date)
            
            if not series_data.empty:
                all_series[series_name] = series_data
            
            time.sleep(0.5)  # Be conservative with rate limiting
        
        if all_series:
            combined_df = pd.DataFrame(all_series)
            combined_df = combined_df.fillna(method='ffill')
            
            logger.info(f"Combined {len(all_series)} FRED series into DataFrame: {combined_df.shape}")
            return combined_df
        
        logger.warning("No FRED series data collected")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching multiple FRED series: {e}")
        return pd.DataFrame()

def parse_api_error(response: requests.Response) -> str:
    """Parse API error response and return meaningful error message"""
    try:
        error_msg = f"API Error {response.status_code}"
        
        try:
            error_data = response.json()
            if 'error_message' in error_data:
                error_msg += f": {error_data['error_message']}"
            elif 'message' in error_data:
                error_msg += f": {error_data['message']}"
            elif 'error' in error_data:
                error_msg += f": {error_data['error']}"
        except:
            error_msg += f": {response.text[:200]}"
        
        return error_msg
        
    except Exception as e:
        return f"Error parsing API error: {e}"

def check_api_status(base_url: str, timeout: int = 10) -> Dict[str, Any]:
    """Check API availability and response time"""
    try:
        start_time = time.time()
        
        try:
            response = requests.get(base_url, timeout=timeout)
            response_time = time.time() - start_time
            
            status = {
                'available': response.status_code == 200,
                'status_code': response.status_code,
                'response_time_ms': round(response_time * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            if response.status_code != 200:
                status['error'] = parse_api_error(response)
            
        except requests.exceptions.Timeout:
            status = {
                'available': False,
                'error': f'Timeout after {timeout}s',
                'response_time_ms': timeout * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            status = {
                'available': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking API status: {e}")
        return {
            'available': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def format_api_data_for_dataframe(api_data: Dict, date_field: str = 'date', 
                                 value_field: str = 'value') -> pd.DataFrame:
    """Format generic API data into pandas DataFrame"""
    try:
        if not api_data or 'data' not in api_data:
            return pd.DataFrame()
        
        data_list = api_data['data']
        
        if not isinstance(data_list, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        
        if date_field in df.columns:
            df[date_field] = pd.to_datetime(df[date_field])
            df = df.set_index(date_field)
        
        # Convert numeric columns
        for col in df.columns:
            if col != date_field:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Formatted API data to DataFrame: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error formatting API data: {e}")
        return pd.DataFrame()

def create_api_request_config(base_url: str, endpoint: str = "", 
                             params: Dict = None, headers: Dict = None) -> Dict:
    """Create standardized API request configuration"""
    try:
        config = {
            'url': base_url.rstrip('/') + '/' + endpoint.lstrip('/') if endpoint else base_url,
            'params': params or {},
            'headers': headers or {},
            'timeout': 30
        }
        
        # Add common headers
        config['headers'].update({
            'User-Agent': 'Treasury-Analytics/1.0',
            'Accept': 'application/json'
        })
        
        return config
        
    except Exception as e:
        logger.error(f"Error creating API request config: {e}")
        return {}

def validate_date_format(date_string: str, expected_format: str = '%Y-%m-%d') -> bool:
    """Validate date string format"""
    try:
        datetime.strptime(date_string, expected_format)
        return True
    except ValueError:
        return False

def get_api_usage_stats(api_calls_log: List[Dict]) -> Dict:
    """Calculate API usage statistics"""
    try:
        if not api_calls_log:
            return {}
        
        total_calls = len(api_calls_log)
        
        # Calculate time range
        timestamps = [call.get('timestamp') for call in api_calls_log if call.get('timestamp')]
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_range_hours = (max_time - min_time).total_seconds() / 3600
        else:
            time_range_hours = 0
        
        # Success rate
        successful_calls = sum(1 for call in api_calls_log if call.get('success', False))
        success_rate = successful_calls / total_calls if total_calls > 0 else 0
        
        # Average response time
        response_times = [call.get('response_time', 0) for call in api_calls_log if call.get('response_time')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        stats = {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': total_calls - successful_calls,
            'success_rate': success_rate,
            'time_range_hours': time_range_hours,
            'calls_per_hour': total_calls / time_range_hours if time_range_hours > 0 else 0,
            'avg_response_time_ms': round(avg_response_time, 2)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating API usage stats: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("API utils module loaded successfully")
    print("Available functions:")
    print("- make_api_request")
    print("- fetch_fred_data")
    print("- fetch_treasury_data")
    print("- fetch_ecb_data")
    print("- batch_api_requests")
    print("- fetch_multiple_fred_series")
    print("- check_api_status")
    print("- cache_api_response")
    print("- load_cached_response")
    print("- APIRateLimiter class")