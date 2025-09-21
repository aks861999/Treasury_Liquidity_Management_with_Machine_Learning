import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def calculate_current_metrics(market_data: pd.DataFrame, customer_data: Dict) -> Dict:
    """Calculate current key metrics for dashboard"""
    try:
        metrics = {}
        
        # Market-based metrics
        if not market_data.empty:
            latest_market = market_data.iloc[-1]
            
            metrics.update({
                'fed_funds_rate': latest_market.get('FED_FUNDS_RATE', 2.0),
                'treasury_10y': latest_market.get('TREASURY_10Y', 3.0),
                'vix_level': latest_market.get('VIX', 20.0),
                'yield_spread': latest_market.get('TREASURY_10Y', 3.0) - latest_market.get('TREASURY_2Y', 2.0)
            })
        
        # Customer-based metrics
        customers = customer_data.get('customers', pd.DataFrame())
        flows = customer_data.get('flows', pd.DataFrame())
        
        if not customers.empty:
            metrics.update({
                'total_customers': len(customers),
                'total_deposits': customers['balance_avg'].sum() if 'balance_avg' in customers.columns else 0,
                'avg_customer_balance': customers['balance_avg'].mean() if 'balance_avg' in customers.columns else 0,
                'customer_segments': len(customers['segment'].unique()) if 'segment' in customers.columns else 0
            })
        
        if not flows.empty:
            metrics.update({
                'daily_flow_volume': flows['deposit_flow'].sum() if 'deposit_flow' in flows.columns else 0,
                'flow_volatility': flows['deposit_flow'].std() if 'deposit_flow' in flows.columns else 0,
                'active_flow_customers': flows['customer_id'].nunique() if 'customer_id' in flows.columns else 0
            })
        
        # Risk metrics (simplified calculations)
        metrics.update({
            'lcr_ratio': calculate_lcr_estimate(metrics),
            'deposit_concentration': calculate_concentration_metric(customers),
            'behavioral_risk_score': calculate_behavioral_risk(customers)
        })
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating current metrics: {e}")
        return {}

def calculate_lcr_estimate(metrics: Dict) -> float:
    """Estimate LCR ratio based on available data"""
    try:
        # Simplified LCR calculation
        total_deposits = metrics.get('total_deposits', 1000000000)
        liquid_assets = total_deposits * 0.15  # Assume 15% liquid assets
        net_outflows = total_deposits * 0.1    # Assume 10% potential outflows
        
        lcr_ratio = liquid_assets / (net_outflows + 1e-8)
        
        # Add some randomness for demonstration
        lcr_ratio *= np.random.uniform(0.95, 1.05)
        
        return max(0.8, min(1.5, lcr_ratio))  # Bound between 80% and 150%
        
    except Exception:
        return 1.15  # Default reasonable value

def calculate_concentration_metric(customers: pd.DataFrame) -> float:
    """Calculate deposit concentration risk metric"""
    try:
        if customers.empty or 'balance_avg' not in customers.columns:
            return 0.3  # Default concentration
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        total_deposits = customers['balance_avg'].sum()
        if total_deposits == 0:
            return 0.3
        
        market_shares = customers['balance_avg'] / total_deposits
        hhi = (market_shares ** 2).sum()
        
        return min(1.0, hhi * 10)  # Scale to 0-1 range
        
    except Exception:
        return 0.3

def calculate_behavioral_risk(customers: pd.DataFrame) -> float:
    """Calculate aggregated behavioral risk score"""
    try:
        if customers.empty:
            return 0.5  # Default medium risk
        
        behavioral_cols = ['rate_sensitivity', 'volatility_factor']
        available_cols = [col for col in behavioral_cols if col in customers.columns]
        
        if not available_cols:
            return 0.5
        
        # Weight by balance
        if 'balance_avg' in customers.columns:
            weights = customers['balance_avg'] / customers['balance_avg'].sum()
            risk_scores = customers[available_cols].mean(axis=1)
            weighted_risk = (risk_scores * weights).sum()
        else:
            weighted_risk = customers[available_cols].mean().mean()
        
        return min(1.0, max(0.0, weighted_risk))
        
    except Exception:
        return 0.5

def format_metric_delta(current: float, previous: float, format_type: str = 'percentage') -> str:
    """Format metric delta for display"""
    try:
        if previous == 0:
            return "N/A"
        
        delta = current - previous
        delta_pct = (delta / previous) * 100
        
        if format_type == 'percentage':
            return f"{delta_pct:+.1f}%"
        elif format_type == 'absolute':
            return f"{delta:+.2f}"
        elif format_type == 'currency':
            return f"${delta:+,.0f}"
        else:
            return f"{delta:+.2f}"
            
    except Exception:
        return "N/A"

def create_performance_summary(models_data: Dict) -> Dict:
    """Create performance summary for models"""
    try:
        if not models_data or 'individual_models' not in models_data:
            return {}
        
        individual_models = models_data['individual_models']
        performance_data = individual_models.get('performance_comparison', pd.DataFrame())
        
        if performance_data.empty:
            return {}
        
        # Best model
        best_model = performance_data.iloc[0]
        
        summary = {
            'best_model_name': best_model.get('model_name', 'Unknown'),
            'best_rmse': best_model.get('rmse', 0),
            'best_r2': best_model.get('r2_score', 0),
            'best_mae': best_model.get('mae', 0),
            'total_models': len(performance_data),
            'avg_r2': performance_data['r2_score'].mean() if 'r2_score' in performance_data.columns else 0
        }
        
        return summary
        
    except Exception as e:
        return {}

def display_metric_card(title: str, value: str, delta: str = None, 
                       help_text: str = None, color: str = "blue"):
    """Display a metric card with consistent styling"""
    try:
        # Color mapping
        colors = {
            'blue': '#1f77b4',
            'green': '#2ca02c', 
            'orange': '#ff7f0e',
            'red': '#d62728',
            'purple': '#9467bd'
        }
        
        border_color = colors.get(color, '#1f77b4')
        
        # Build delta HTML
        delta_html = ""
        if delta and delta != "N/A":
            delta_color = "green" if delta.startswith('+') or not delta.startswith('-') else "red"
            delta_html = f"<div style='color: {delta_color}; font-size: 0.8rem; margin-top: 0.2rem;'>{delta}</div>"
        
        # Build help HTML
        help_html = ""
        if help_text:
            help_html = f"<div style='color: #666; font-size: 0.7rem; margin-top: 0.3rem;'>{help_text}</div>"
        
        # Complete card HTML
        card_html = f"""
        <div style='
            background: white; 
            padding: 1rem; 
            border-radius: 8px; 
            border-left: 4px solid {border_color}; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        '>
            <div style='font-size: 0.85rem; color: #666; margin-bottom: 0.3rem;'>{title}</div>
            <div style='font-size: 1.5rem; font-weight: bold; color: #333;'>{value}</div>
            {delta_html}
            {help_html}
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying metric card: {e}")

def create_metrics_grid(metrics: Dict, layout: List[List[str]]):
    """Create a grid layout of metrics"""
    try:
        for row in layout:
            cols = st.columns(len(row))
            
            for i, metric_key in enumerate(row):
                if metric_key in metrics:
                    with cols[i]:
                        metric_info = metrics[metric_key]
                        display_metric_card(
                            title=metric_info.get('title', metric_key),
                            value=metric_info.get('value', 'N/A'),
                            delta=metric_info.get('delta'),
                            help_text=metric_info.get('help'),
                            color=metric_info.get('color', 'blue')
                        )
                else:
                    with cols[i]:
                        st.info(f"Metric '{metric_key}' not available")
        
    except Exception as e:
        st.error(f"Error creating metrics grid: {e}")

def format_large_number(value: float, precision: int = 1) -> str:
    """Format large numbers with appropriate suffixes"""
    try:
        if abs(value) >= 1e12:
            return f"{value/1e12:.{precision}f}T"
        elif abs(value) >= 1e9:
            return f"{value/1e9:.{precision}f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.{precision}f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
            
    except Exception:
        return str(value)

def calculate_trend_indicator(series: pd.Series, window: int = 5) -> str:
    """Calculate trend indicator for a time series"""
    try:
        if len(series) < window:
            return "stable"
        
        recent_trend = series.tail(window).diff().mean()
        overall_std = series.std()
        
        threshold = overall_std * 0.1  # 10% of standard deviation
        
        if recent_trend > threshold:
            return "increasing"
        elif recent_trend < -threshold:
            return "decreasing" 
        else:
            return "stable"
            
    except Exception:
        return "stable"

def create_status_indicator(value: float, thresholds: Dict[str, float]) -> Tuple[str, str]:
    """Create status indicator based on thresholds"""
    try:
        # Default thresholds
        default_thresholds = {
            'excellent': 0.9,
            'good': 0.7, 
            'fair': 0.5,
            'poor': 0.3
        }
        
        thresholds = {**default_thresholds, **thresholds}
        
        if value >= thresholds['excellent']:
            return "🟢", "Excellent"
        elif value >= thresholds['good']:
            return "🔵", "Good"
        elif value >= thresholds['fair']:
            return "🟡", "Fair"
        elif value >= thresholds['poor']:
            return "🟠", "Poor"
        else:
            return "🔴", "Critical"
            
    except Exception:
        return "⚪", "Unknown"

def display_kpi_dashboard(metrics: Dict):
    """Display comprehensive KPI dashboard"""
    try:
        # Define KPI structure
        kpi_structure = {
            'Financial Health': [
                {
                    'key': 'lcr_ratio',
                    'title': 'LCR Ratio',
                    'format': 'percentage',
                    'thresholds': {'excellent': 1.2, 'good': 1.1, 'fair': 1.0, 'poor': 0.9},
                    'help': 'Basel III Liquidity Coverage Ratio'
                },
                {
                    'key': 'total_deposits', 
                    'title': 'Total Deposits',
                    'format': 'currency',
                    'help': 'Total customer deposits across all segments'
                }
            ],
            'Market Environment': [
                {
                    'key': 'fed_funds_rate',
                    'title': 'Fed Funds Rate', 
                    'format': 'percentage',
                    'help': 'Federal funds rate affecting deposit pricing'
                },
                {
                    'key': 'vix_level',
                    'title': 'Market Volatility (VIX)',
                    'format': 'number',
                    'thresholds': {'excellent': 15, 'good': 20, 'fair': 25, 'poor': 30},
                    'help': 'Market volatility indicator'
                }
            ],
            'Customer Analytics': [
                {
                    'key': 'total_customers',
                    'title': 'Total Customers',
                    'format': 'number',
                    'help': 'Active customer count'
                },
                {
                    'key': 'behavioral_risk_score',
                    'title': 'Behavioral Risk Score',
                    'format': 'percentage', 
                    'thresholds': {'excellent': 0.3, 'good': 0.5, 'fair': 0.7, 'poor': 0.8},
                    'help': 'Aggregated customer behavioral risk'
                }
            ]
        }
        
        # Display KPI sections
        for section_name, kpis in kpi_structure.items():
            st.subheader(f"📊 {section_name}")
            
            cols = st.columns(len(kpis))
            
            for i, kpi in enumerate(kpis):
                with cols[i]:
                    key = kpi['key']
                    value = metrics.get(key, 0)
                    
                    # Format value
                    if kpi['format'] == 'percentage':
                        formatted_value = f"{value:.1%}" if isinstance(value, float) else f"{value:.1f}%"
                    elif kpi['format'] == 'currency':
                        formatted_value = f"${format_large_number(value)}"
                    else:
                        formatted_value = f"{format_large_number(value)}"
                    
                    # Get status indicator
                    status_emoji, status_text = "⚪", "Normal"
                    if 'thresholds' in kpi:
                        status_emoji, status_text = create_status_indicator(value, kpi['thresholds'])
                    
                    # Display metric with status
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e0e0e0;'>
                        <div style='font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;'>{kpi['title']}</div>
                        <div style='font-size: 1.8rem; font-weight: bold; color: #333; margin-bottom: 0.3rem;'>{formatted_value}</div>
                        <div style='font-size: 0.9rem;'>{status_emoji} {status_text}</div>
                        <div style='font-size: 0.7rem; color: #999; margin-top: 0.5rem;'>{kpi.get('help', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying KPI dashboard: {e}")

def create_comparison_metrics(current_metrics: Dict, previous_metrics: Dict = None) -> Dict:
    """Create comparison metrics with deltas"""
    try:
        if previous_metrics is None:
            # Generate mock previous metrics for demonstration
            previous_metrics = {}
            for key, value in current_metrics.items():
                if isinstance(value, (int, float)):
                    # Add some variation for demo
                    variation = np.random.uniform(-0.05, 0.05)  # ±5% variation
                    previous_metrics[key] = value * (1 + variation)
        
        comparison_metrics = {}
        
        for key, current_value in current_metrics.items():
            if isinstance(current_value, (int, float)):
                previous_value = previous_metrics.get(key, current_value)
                
                comparison_metrics[key] = {
                    'current': current_value,
                    'previous': previous_value,
                    'delta_abs': current_value - previous_value,
                    'delta_pct': ((current_value - previous_value) / (previous_value + 1e-8)) * 100,
                    'trend': 'up' if current_value > previous_value else 'down' if current_value < previous_value else 'stable'
                }
            else:
                comparison_metrics[key] = {
                    'current': current_value,
                    'previous': previous_metrics.get(key, current_value),
                    'delta_abs': 0,
                    'delta_pct': 0,
                    'trend': 'stable'
                }
        
        return comparison_metrics
        
    except Exception as e:
        st.error(f"Error creating comparison metrics: {e}")
        return {}

if __name__ == "__main__":
    # Test the metrics component
    st.title("Metrics Component Test")
    
    # Sample data
    sample_market_data = pd.DataFrame({
        'FED_FUNDS_RATE': [2.0, 2.1, 2.0],
        'TREASURY_10Y': [3.0, 3.1, 3.0],
        'VIX': [20.0, 22.0, 21.0]
    })
    
    sample_customer_data = {
        'customers': pd.DataFrame({
            'balance_avg': [10000, 50000, 100000],
            'segment': ['RETAIL_SMALL', 'RETAIL_MEDIUM', 'RETAIL_LARGE'],
            'rate_sensitivity': [0.3, 0.5, 0.7]
        })
    }
    
    metrics = calculate_current_metrics(sample_market_data, sample_customer_data)
    st.write("Calculated metrics:", metrics)
    
    display_kpi_dashboard(metrics)