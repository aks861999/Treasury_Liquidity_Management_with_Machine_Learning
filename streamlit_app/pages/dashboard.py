import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import VIZ_CONFIG
from streamlit_app.components.charts import (
    display_key_metrics_cards, create_deposit_flow_chart, 
    create_segment_distribution_chart, create_yield_curve_chart
)
from streamlit_app.components.metrics import (
    calculate_current_metrics, format_metric_delta, 
    create_performance_summary
)

def display_dashboard():
    """Main dashboard page"""
    try:
        # Dashboard header
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🏦 Treasury Behavioral Deposit Analytics</h1>
            <p style='color: white; opacity: 0.9; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Real-time Risk Management & Liquidity Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if data is loaded
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ No data loaded. Please use the sidebar to load data first.")
            
            # Quick start guide
            with st.expander("📚 Quick Start Guide", expanded=True):
                st.markdown("""
                **Get started in 3 easy steps:**
                
                1. 🔄 **Load Data**: Click "Load Data" in the sidebar to collect market data and generate customer data
                2. 🎯 **Train Models**: Once data is loaded, click "Train Models" to build ML models  
                3. 📊 **Explore**: Navigate through different pages to explore insights and analytics
                
                **What you'll see:**
                - Real-time market conditions and key performance indicators
                - Customer behavior analysis and segmentation insights  
                - Advanced risk metrics including LCR ratio and stress testing
                - ML model predictions and performance analytics
                """)
            
            return
        
        # Get data from session state
        aggregated_data = st.session_state.aggregated_data
        market_data = aggregated_data.get('market_data', pd.DataFrame())
        customer_data = aggregated_data.get('customer_data', {})
        
        # Calculate current metrics
        current_metrics = calculate_current_metrics(market_data, customer_data)
        
        # Key Performance Indicators
        st.subheader("📊 Key Performance Indicators")
        display_key_metrics_cards(current_metrics)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Deposit flow analysis
            st.subheader("💸 Deposit Flow Analysis")
            
            unified_ts = aggregated_data.get('unified_timeseries', pd.DataFrame())
            flows = customer_data.get('flows', pd.DataFrame())
            
            if not unified_ts.empty:
                create_deposit_flow_chart(unified_ts)
            elif not flows.empty:
                create_deposit_flow_chart(flows)
            else:
                st.info("No deposit flow data available")
            
            # Add flow statistics
            if not flows.empty:
                with st.expander("📈 Flow Statistics", expanded=False):
                    col1_exp, col2_exp = st.columns(2)
                    
                    with col1_exp:
                        st.metric("Total Daily Flows", f"${flows['deposit_flow'].sum():,.0f}")
                        st.metric("Average Flow", f"${flows['deposit_flow'].mean():,.0f}")
                    
                    with col2_exp:
                        st.metric("Flow Volatility", f"${flows['deposit_flow'].std():,.0f}")
                        st.metric("Active Customers", f"{flows['customer_id'].nunique():,}")
        
        with col2:
            # Current market conditions
            st.subheader("🌡️ Market Environment")
            
            if not market_data.empty:
                # Market condition indicators
                latest_market = market_data.iloc[-1]
                
                # Interest rate environment
                fed_rate = latest_market.get('FED_FUNDS_RATE', 2.0)
                rate_color = "🔴" if fed_rate > 5 else "🟡" if fed_rate > 2 else "🟢"
                st.write(f"{rate_color} **Fed Funds Rate**: {fed_rate:.2f}%")
                
                # Market stress level
                vix = latest_market.get('VIX', 20)
                stress_color = "🔴" if vix > 30 else "🟡" if vix > 20 else "🟢"
                stress_level = "High" if vix > 30 else "Medium" if vix > 20 else "Low"
                st.write(f"{stress_color} **Market Stress**: {stress_level} (VIX: {vix:.1f})")
                
                # Yield curve
                if all(col in market_data.columns for col in ['TREASURY_10Y', 'TREASURY_2Y']):
                    spread = latest_market['TREASURY_10Y'] - latest_market['TREASURY_2Y']
                    curve_color = "🔴" if spread < 0 else "🟡" if spread < 0.5 else "🟢"
                    curve_shape = "Inverted" if spread < 0 else "Flat" if spread < 0.5 else "Normal"
                    st.write(f"{curve_color} **Yield Curve**: {curve_shape} ({spread:.1f}bps)")
                
                # Add mini yield curve chart
                create_yield_curve_chart(market_data)
            else:
                st.info("No market data available")
            
            # Risk alerts
            st.subheader("⚠️ Risk Alerts")
            
            alerts = []
            
            # LCR alert
            lcr_ratio = current_metrics.get('lcr_ratio', 1.15)
            if lcr_ratio < 1.0:
                alerts.append("🔴 **CRITICAL**: LCR below regulatory minimum")
            elif lcr_ratio < 1.1:
                alerts.append("🟡 **WARNING**: LCR below buffer threshold")
            
            # Market stress alert
            if vix > 30:
                alerts.append("🔴 **HIGH STRESS**: Elevated market volatility")
            
            # Flow volatility alert
            if not flows.empty:
                flow_vol = flows['deposit_flow'].std()
                if flow_vol > flows['deposit_flow'].mean() * 0.5:
                    alerts.append("🟡 **ATTENTION**: High deposit flow volatility")
            
            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ **ALL CLEAR**: No immediate risk alerts")
        
        st.markdown("---")
        
        # Bottom section - Performance & Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Customer Segments")
            
            customers = customer_data.get('customers', pd.DataFrame())
            if not customers.empty:
                create_segment_distribution_chart(customers)
                
                # Segment metrics
                segment_stats = customers.groupby('segment').agg({
                    'balance_avg': ['count', 'sum', 'mean'],
                    'rate_sensitivity': 'mean',
                    'loyalty_score': 'mean'
                }).round(2)
                
                # Flatten column names
                segment_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in segment_stats.columns]
                
                with st.expander("📊 Segment Details", expanded=False):
                    st.dataframe(segment_stats, use_container_width=True)
            else:
                st.info("No customer data available")
        
        with col2:
            st.subheader("🤖 Model Performance")
            
            if st.session_state.get('models_trained', False):
                models = st.session_state.models
                performance_data = models.get('individual_models', {}).get('performance_comparison', pd.DataFrame())
                
                if not performance_data.empty:
                    # Best model summary
                    best_model = performance_data.iloc[0]
                    
                    st.success(f"**Best Model**: {best_model['model_name']}")
                    
                    col1_model, col2_model = st.columns(2)
                    with col1_model:
                        st.metric("RMSE", f"{best_model['rmse']:.4f}")
                        st.metric("R² Score", f"{best_model['r2_score']:.4f}")
                    
                    with col2_model:
                        st.metric("MAE", f"{best_model['mae']:.4f}")
                        st.metric("MAPE", f"{best_model.get('mape', 0):.2f}%")
                    
                    # Model comparison chart
                    fig = px.bar(
                        performance_data.head(3),  # Top 3 models
                        x='model_name',
                        y='r2_score',
                        title="Model R² Comparison",
                        color='r2_score',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Model performance data not available")
            else:
                st.info("No models trained yet. Use the sidebar to train models.")
                
                if st.button("🚀 Quick Train Models"):
                    # This would trigger model training
                    st.info("Model training would be initiated here...")
        
        # Footer with timestamp
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
                f"📊 Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
                unsafe_allow_html=True
            )
        
    except Exception as e:
        st.error(f"Error displaying dashboard: {e}")
        st.info("Please try refreshing the page or contact support.")

def create_summary_cards():
    """Create summary cards for quick overview"""
    try:
        if not st.session_state.get('data_loaded', False):
            return
        
        # Get summary statistics
        aggregated_data = st.session_state.aggregated_data
        market_data = aggregated_data.get('market_data', pd.DataFrame())
        customer_data = aggregated_data.get('customer_data', {})
        
        # Create metrics
        metrics = []
        
        if not market_data.empty:
            metrics.append({
                'title': 'Market Data Points',
                'value': len(market_data),
                'icon': '📈'
            })
        
        customers = customer_data.get('customers', pd.DataFrame())
        if not customers.empty:
            metrics.append({
                'title': 'Total Customers',
                'value': len(customers),
                'icon': '👥'
            })
        
        flows = customer_data.get('flows', pd.DataFrame())
        if not flows.empty:
            metrics.append({
                'title': 'Flow Records',
                'value': len(flows),
                'icon': '💸'
            })
        
        # Display metrics in columns
        cols = st.columns(len(metrics))
        for i, metric in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 10px; text-align: center; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{metric['icon']}</div>
                    <div style='font-size: 1.5rem; font-weight: bold; color: #333;'>{metric['value']:,}</div>
                    <div style='color: #666; font-size: 0.9rem;'>{metric['title']