import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import VIZ_CONFIG
from src.utils.visualization_utils import (
    create_time_series_plot, create_correlation_heatmap, 
    create_distribution_plot, create_bar_chart, create_pie_chart,
    create_gauge_chart, create_waterfall_chart, create_model_performance_plot
)

def display_key_metrics_cards(metrics: Dict):
    """Display key metrics in card format"""
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            lcr_value = metrics.get('lcr_ratio', 1.15)
            lcr_delta = f"{(lcr_value - 1.0) * 100:+.1f}%" if lcr_value != 1.0 else "0%"
            st.metric(
                label="🛡️ LCR Ratio",
                value=f"{lcr_value:.1%}",
                delta=lcr_delta,
                help="Liquidity Coverage Ratio - Basel III requirement"
            )
        
        with col2:
            deposits_value = metrics.get('total_deposits', 1000000000)
            deposits_delta = f"{np.random.uniform(-2, 5):.1f}%"  # Mock delta
            st.metric(
                label="💰 Total Deposits",
                value=f"${deposits_value/1e9:.1f}B",
                delta=deposits_delta,
                help="Total customer deposits across all segments"
            )
        
        with col3:
            rate_value = metrics.get('fed_funds_rate', 2.5)
            rate_delta = f"{np.random.uniform(-0.25, 0.25):+.2f}%"  # Mock delta
            st.metric(
                label="📈 Fed Funds Rate",
                value=f"{rate_value:.2f}%",
                delta=rate_delta,
                help="Federal funds rate affecting deposit pricing"
            )
        
        with col4:
            vix_value = metrics.get('vix_level', 20)
            stress_level = "Low" if vix_value < 20 else "Medium" if vix_value < 30 else "High"
            st.metric(
                label="⚠️ Market Stress",
                value=f"{vix_value:.1f}",
                delta=stress_level,
                help="VIX level indicating market volatility"
            )
    
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

def create_deposit_flow_chart(data: pd.DataFrame):
    """Create deposit flow time series chart"""
    try:
        if data.empty or 'date' not in data.columns:
            st.info("No deposit flow data available")
            return
        
        # Prepare data
        chart_data = data.copy()
        chart_data['date'] = pd.to_datetime(chart_data['date'])
        
        # Get flow columns
        flow_cols = [col for col in chart_data.columns if 'flow' in col.lower()]
        if not flow_cols:
            flow_cols = [col for col in chart_data.columns if col not in ['date', 'segment']]
        
        if flow_cols:
            fig = create_time_series_plot(
                chart_data, 'date', flow_cols[:3],  # Limit to first 3 columns
                title="Daily Deposit Flows"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flow columns found in data")
    
    except Exception as e:
        st.error(f"Error creating deposit flow chart: {e}")

def create_segment_distribution_chart(customer_data: pd.DataFrame):
    """Create customer segment distribution chart"""
    try:
        if customer_data.empty or 'segment' not in customer_data.columns:
            st.info("No customer segment data available")
            return
        
        # Calculate segment distribution
        segment_counts = customer_data['segment'].value_counts()
        
        # Create pie chart
        fig = create_pie_chart(
            pd.DataFrame({
                'segment': segment_counts.index,
                'count': segment_counts.values
            }),
            'count', 'segment',
            title="Customer Distribution by Segment"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating segment distribution chart: {e}")

def create_correlation_chart(data: pd.DataFrame):
    """Create correlation heatmap"""
    try:
        if data.empty:
            st.info("No data available for correlation analysis")
            return
        
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation")
            return
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create heatmap
        fig = create_correlation_heatmap(corr_matrix, "Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating correlation chart: {e}")

def create_model_comparison_chart(performance_data: pd.DataFrame):
    """Create model performance comparison chart"""
    try:
        if performance_data.empty:
            st.info("No model performance data available")
            return
        
        # Create comparison chart
        fig = create_bar_chart(
            performance_data,
            'model_name', 'rmse',
            title="Model Performance Comparison (RMSE)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display performance table
        st.subheader("Detailed Performance Metrics")
        display_cols = ['model_name', 'rmse', 'r2_score', 'mape']
        available_cols = [col for col in display_cols if col in performance_data.columns]
        
        if available_cols:
            st.dataframe(
                performance_data[available_cols].round(4),
                use_container_width=True,
                hide_index=True
            )
    
    except Exception as e:
        st.error(f"Error creating model comparison chart: {e}")

def create_feature_importance_chart(feature_importance: pd.DataFrame, top_n: int = 15):
    """Create feature importance chart"""
    try:
        if feature_importance.empty:
            st.info("No feature importance data available")
            return
        
        # Take top N features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(color=VIZ_CONFIG['COLORS']['PRIMARY'])
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating feature importance chart: {e}")

def create_risk_gauge_chart(risk_value: float, title: str = "Risk Level"):
    """Create risk gauge chart"""
    try:
        # Create gauge chart
        fig = create_gauge_chart(
            value=risk_value,
            min_val=0,
            max_val=10,
            title=title
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating risk gauge chart: {e}")

def create_scenario_waterfall_chart(base_value: float, scenarios: Dict[str, float]):
    """Create scenario impact waterfall chart"""
    try:
        # Prepare waterfall data
        categories = ['Base Case']
        values = [base_value]
        
        for scenario, value in scenarios.items():
            categories.append(scenario)
            values.append(value - base_value)
        
        categories.append('Final Value')
        values.append(sum(values))
        
        # Create waterfall chart
        fig = create_waterfall_chart(categories, values, "Scenario Impact Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating waterfall chart: {e}")

def create_behavioral_analysis_chart(customer_data: pd.DataFrame):
    """Create behavioral analysis charts"""
    try:
        if customer_data.empty:
            st.info("No customer data available for behavioral analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rate sensitivity by segment
            if 'rate_sensitivity' in customer_data.columns and 'segment' in customer_data.columns:
                avg_sensitivity = customer_data.groupby('segment')['rate_sensitivity'].mean().reset_index()
                
                fig = create_bar_chart(
                    avg_sensitivity,
                    'segment', 'rate_sensitivity',
                    title="Average Rate Sensitivity by Segment"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Rate sensitivity data not available")
        
        with col2:
            # Loyalty vs Volatility scatter
            if all(col in customer_data.columns for col in ['loyalty_score', 'volatility_factor', 'segment']):
                fig = px.scatter(
                    customer_data,
                    x='loyalty_score',
                    y='volatility_factor',
                    color='segment',
                    size='balance_avg' if 'balance_avg' in customer_data.columns else None,
                    title="Customer Loyalty vs Volatility",
                    color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT']
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Behavioral metrics not available")
    
    except Exception as e:
        st.error(f"Error creating behavioral analysis chart: {e}")

def create_yield_curve_chart(market_data: pd.DataFrame):
    """Create yield curve chart"""
    try:
        if market_data.empty:
            st.info("No market data available for yield curve")
            return
        
        # Look for treasury columns
        treasury_cols = [col for col in market_data.columns if 'TREASURY' in col]
        
        if len(treasury_cols) < 2:
            st.info("Insufficient treasury data for yield curve")
            return
        
        # Get latest values
        latest_data = market_data.iloc[-1]
        
        # Create simple yield curve
        maturities = []
        yields = []
        
        maturity_mapping = {
            'TREASURY_3M': '3M',
            'TREASURY_6M': '6M',
            'TREASURY_1Y': '1Y',
            'TREASURY_2Y': '2Y',
            'TREASURY_5Y': '5Y',
            'TREASURY_10Y': '10Y',
            'TREASURY_30Y': '30Y'
        }
        
        for col in treasury_cols:
            if col in maturity_mapping and not pd.isna(latest_data[col]):
                maturities.append(maturity_mapping[col])
                yields.append(latest_data[col])
        
        if len(maturities) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=maturities,
                y=yields,
                mode='lines+markers',
                name='Yield Curve',
                line=dict(color=VIZ_CONFIG['COLORS']['PRIMARY'], width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Current Yield Curve",
                xaxis_title="Maturity",
                yaxis_title="Yield (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough yield curve data points")
    
    except Exception as e:
        st.error(f"Error creating yield curve chart: {e}")

def create_prediction_accuracy_chart(actual: np.ndarray, predicted: np.ndarray, 
                                   model_name: str = "Model"):
    """Create prediction accuracy visualization"""
    try:
        if len(actual) == 0 or len(predicted) == 0:
            st.info("No prediction data available")
            return
        
        # Create model performance plot
        fig = create_model_performance_plot(actual, predicted, model_name)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("R² Score", f"{r2:.4f}")
    
    except Exception as e:
        st.error(f"Error creating prediction accuracy chart: {e}")

def display_data_quality_metrics(data_quality: Dict):
    """Display data quality metrics"""
    try:
        if not data_quality:
            st.info("No data quality metrics available")
            return
        
        overall_score = data_quality.get('overall_score', 0)
        
        # Quality score gauge
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = create_gauge_chart(
                value=overall_score,
                min_val=0,
                max_val=100,
                title="Data Quality Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality details
            st.subheader("Quality Details")
            
            market_quality = data_quality.get('market_data_quality', {})
            if market_quality:
                st.write(f"**Market Data Records**: {market_quality.get('total_records', 'N/A')}")
                st.write(f"**Completeness**: {market_quality.get('completeness_score', 0):.1f}%")
                st.write(f"**Missing Values**: {market_quality.get('missing_values_pct', 0):.1f}%")
            
            customer_quality = data_quality.get('customer_data_quality', {})
            if customer_quality:
                st.write(f"**Total Customers**: {customer_quality.get('total_customers', 'N/A')}")
                st.write(f"**Flow Records**: {customer_quality.get('total_flow_records', 'N/A')}")
        
        # Issues and recommendations
        issues = data_quality.get('issues', [])
        recommendations = data_quality.get('recommendations', [])
        
        if issues:
            st.subheader("Data Quality Issues")
            for issue in issues:
                st.warning(f"⚠️ {issue}")
        
        if recommendations:
            st.subheader("Recommendations")
            for rec in recommendations:
                st.info(f"💡 {rec}")
    
    except Exception as e:
        st.error(f"Error displaying data quality metrics: {e}")

if __name__ == "__main__":
    st.title("Chart Components Test")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'deposit_flow': np.random.randn(100) * 1000000,
        'rate': np.random.randn(100) * 0.01 + 0.02
    })
    
    st.subheader("Sample Deposit Flow Chart")
    create_deposit_flow_chart(sample_data)