import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import STREAMLIT_CONFIG, VIZ_CONFIG
from src.data_collection.data_aggregator import aggregate_all_data, refresh_data_if_stale
from src.data_preprocessing.feature_engineer import create_feature_engineering_pipeline
from src.models.behavioral_models import main_model_training_pipeline
from src.models.ensemble_model import main_ensemble_pipeline
from streamlit_app.components.charts import (
    display_key_metrics_cards, create_deposit_flow_chart, 
    create_segment_distribution_chart, create_model_comparison_chart,
    create_behavioral_analysis_chart, create_yield_curve_chart,
    display_data_quality_metrics
)

# Update imports in main_app.py
from streamlit_app.pages.dashboard import display_dashboard
from streamlit_app.pages.market_analysis import display_market_analysis  
from streamlit_app.pages.behavioral_modeling import display_behavioral_modeling
from streamlit_app.pages.scenario_analysis import display_scenario_analysis
from streamlit_app.pages.model_performance import display_model_performance


# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG['PAGE_TITLE'],
    page_icon=STREAMLIT_CONFIG['PAGE_ICON'],
    layout=STREAMLIT_CONFIG['LAYOUT'],
    initial_sidebar_state=STREAMLIT_CONFIG['SIDEBAR_STATE']
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.sidebar-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    margin-bottom: 1rem;
}

.stAlert > div {
    padding: 1rem;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'aggregated_data' not in st.session_state:
        st.session_state.aggregated_data = {}
    if 'feature_data' not in st.session_state:
        st.session_state.feature_data = pd.DataFrame()
    if 'models' not in st.session_state:
        st.session_state.models = {}

def load_data():
    """Load and prepare data"""
    try:
        with st.spinner("🔄 Collecting and aggregating data..."):
            # Use data aggregator to get all data
            aggregated_data = refresh_data_if_stale(max_age_hours=6)
            
            if not aggregated_data or aggregated_data.get('market_data', pd.DataFrame()).empty:
                st.error("❌ Failed to collect market data")
                return False
            
            st.session_state.aggregated_data = aggregated_data
            
            # Extract components
            market_data = aggregated_data.get('market_data', pd.DataFrame())
            customer_data = aggregated_data.get('customer_data', {})
            
            # Feature engineering
            if not market_data.empty and customer_data:
                customers = customer_data.get('customers', pd.DataFrame())
                flows = customer_data.get('flows', pd.DataFrame())
                
                if not customers.empty and not flows.empty:
                    feature_data = create_feature_engineering_pipeline(
                        market_data, customers, flows
                    )
                    
                    if not feature_data.empty:
                        st.session_state.feature_data = feature_data
                        st.session_state.data_loaded = True
                        st.success("✅ Data loaded and processed successfully!")
                        return True
        
        st.error("❌ Failed to process data")
        return False
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return False

def train_models():
    """Train machine learning models"""
    if not st.session_state.data_loaded:
        st.error("❌ Please load data first")
        return False
    
    try:
        feature_data = st.session_state.feature_data
        
        if feature_data.empty:
            st.error("❌ No feature data available")
            return False
        
        # Prepare data for modeling
        target_col = 'target_deposit_flow_1d'
        
        # Check if target column exists
        if target_col not in feature_data.columns:
            # Use a proxy target if the exact target doesn't exist
            flow_cols = [col for col in feature_data.columns if 'flow' in col.lower()]
            if flow_cols:
                target_col = flow_cols[0]
            else:
                st.error("❌ No suitable target column found")
                return False
        
        feature_cols = [
            col for col in feature_data.columns 
            if not col.startswith('target_') and col not in ['date', 'segment', 'customer_id']
        ]
        
        if len(feature_cols) < 5:
            st.error("❌ Insufficient features for modeling")
            return False
        
        # Split data temporally
        split_point = int(len(feature_data) * 0.8)
        train_data = feature_data.iloc[:split_point]
        test_data = feature_data.iloc[split_point:]
        
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[target_col].fillna(0)
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[target_col].fillna(0)
        
        with st.spinner("🤖 Training machine learning models..."):
            # Train individual models
            model_results = main_model_training_pipeline(
                X_train, y_train, X_test, y_test,
                hyperparameter_tuning=False
            )
            
            if not model_results:
                st.error("❌ Failed to train models")
                return False
            
            st.session_state.models = {
                'individual_models': model_results,
                'test_data': {'X_test': X_test, 'y_test': y_test}
            }
        
        st.session_state.models_trained = True
        st.success("✅ Models trained successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error training models: {e}")
        return False

def display_main_dashboard():
    """Display main dashboard"""
    st.markdown('<div class="main-header">🏦 Treasury Behavioral Deposit Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data using the sidebar controls")
        st.info("👈 Use the sidebar to get started with data collection and model training")
        return
    
    # Get data
    aggregated_data = st.session_state.aggregated_data
    market_data = aggregated_data.get('market_data', pd.DataFrame())
    customer_data = aggregated_data.get('customer_data', {})
    
    # Display key metrics
    st.subheader("📊 Key Performance Indicators")
    
    # Calculate current metrics
    latest_metrics = {}
    if not market_data.empty:
        latest_data = market_data.iloc[-1]
        latest_metrics = {
            'lcr_ratio': np.random.uniform(1.1, 1.3),  # Mock LCR
            'total_deposits': np.random.uniform(0.8e9, 1.2e9),  # Mock deposits
            'fed_funds_rate': latest_data.get('FED_FUNDS_RATE', 2.5),
            'vix_level': latest_data.get('VIX', 20)
        }
    
    display_key_metrics_cards(latest_metrics)
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Deposit Flow Analysis")
        
        # Use unified time series if available
        unified_ts = aggregated_data.get('unified_timeseries', pd.DataFrame())
        if not unified_ts.empty:
            create_deposit_flow_chart(unified_ts)
        elif customer_data and 'flows' in customer_data:
            flows = customer_data['flows']
            if not flows.empty:
                create_deposit_flow_chart(flows)
        else:
            st.info("No deposit flow data available")
    
    with col2:
        st.subheader("🎯 Customer Segments")
        
        if customer_data and 'customers' in customer_data:
            customers = customer_data['customers']
            if not customers.empty:
                create_segment_distribution_chart(customers)
        else:
            st.info("No customer data available")
    
    # Market analysis section
    st.markdown("---")
    st.subheader("📊 Market Environment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not market_data.empty:
            create_yield_curve_chart(market_data)
        else:
            st.info("No market data available for yield curve")
    
    with col2:
        # Market conditions summary
        if not market_data.empty:
            st.write("**Current Market Conditions**")
            latest = market_data.iloc[-1]
            
            conditions = []
            fed_rate = latest.get('FED_FUNDS_RATE', 0)
            if fed_rate < 1:
                conditions.append("🟢 Low interest rate environment")
            elif fed_rate > 4:
                conditions.append("🔴 High interest rate environment")
            else:
                conditions.append("🟡 Normal interest rate environment")
            
            vix = latest.get('VIX', 20)
            if vix < 20:
                conditions.append("🟢 Low market volatility")
            elif vix > 30:
                conditions.append("🔴 High market volatility")
            else:
                conditions.append("🟡 Moderate market volatility")
            
            for condition in conditions:
                st.write(condition)
        else:
            st.info("No market data available")
    
    # Model performance section (if models trained)
    if st.session_state.models_trained:
        st.markdown("---")
        st.subheader("🤖 Model Performance")
        
        models = st.session_state.models
        individual_models = models.get('individual_models', {})
        performance_data = individual_models.get('performance_comparison', pd.DataFrame())
        
        if not performance_data.empty:
            create_model_comparison_chart(performance_data)
        else:
            st.info("No model performance data available")

def display_data_overview():
    """Display data overview and quality"""
    st.header("📋 Data Overview")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded yet. Please use the sidebar to load data.")
        return
    
    aggregated_data = st.session_state.aggregated_data
    
    # Data summary
    st.subheader("📊 Data Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_data = aggregated_data.get('market_data', pd.DataFrame())
        st.metric(
            "Market Data Records",
            len(market_data) if not market_data.empty else 0
        )
    
    with col2:
        customer_data = aggregated_data.get('customer_data', {})
        customers = customer_data.get('customers', pd.DataFrame())
        st.metric(
            "Customer Records",
            len(customers) if not customers.empty else 0
        )
    
    with col3:
        flows = customer_data.get('flows', pd.DataFrame())
        st.metric(
            "Flow Records",
            len(flows) if not flows.empty else 0
        )
    
    # Data quality
    data_quality = aggregated_data.get('data_quality', {})
    if data_quality:
        st.subheader("🎯 Data Quality Assessment")
        display_data_quality_metrics(data_quality)
    
    # Data samples
    st.subheader("🔍 Data Samples")
    
    tab1, tab2, tab3 = st.tabs(["Market Data", "Customer Data", "Flow Data"])
    
    with tab1:
        if not market_data.empty:
            st.write("**Market Data Sample**")
            st.dataframe(market_data.head(), use_container_width=True)
        else:
            st.info("No market data available")
    
    with tab2:
        if not customers.empty:
            st.write("**Customer Data Sample**")
            st.dataframe(customers.head(), use_container_width=True)
        else:
            st.info("No customer data available")
    
    with tab3:
        if not flows.empty:
            st.write("**Flow Data Sample**")
            st.dataframe(flows.head(), use_container_width=True)
        else:
            st.info("No flow data available")

def display_behavioral_analysis():
    """Display behavioral analysis page"""
    st.header("🧠 Behavioral Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first to view behavioral analysis")
        return
    
    aggregated_data = st.session_state.aggregated_data
    customer_data = aggregated_data.get('customer_data', {})
    
    if not customer_data:
        st.error("No customer data available for behavioral analysis")
        return
    
    customers = customer_data.get('customers', pd.DataFrame())
    flows = customer_data.get('flows', pd.DataFrame())
    
    # Behavioral insights
    st.subheader("📊 Customer Behavior Insights")
    
    if not customers.empty:
        create_behavioral_analysis_chart(customers)
    
    # Segment analysis
    if not customers.empty and not flows.empty:
        st.subheader("🎯 Segment Analysis")
        
        # Flow patterns by segment
        col1, col2 = st.columns(2)
        
        with col1:
            if 'segment' in flows.columns and 'deposit_flow' in flows.columns:
                segment_flows = flows.groupby('segment')['deposit_flow'].agg([
                    'mean', 'std', 'count'
                ]).reset_index()
                
                st.write("**Average Flow by Segment**")
                st.dataframe(segment_flows.round(2), use_container_width=True)
        
        with col2:
            if 'segment' in customers.columns and 'balance_avg' in customers.columns:
                segment_balance = customers.groupby('segment')['balance_avg'].agg([
                    'mean', 'sum', 'count'
                ]).reset_index()
                
                st.write("**Balance Distribution by Segment**")
                st.dataframe(segment_balance.round(2), use_container_width=True)

def display_sidebar():
    """Display sidebar controls"""
    st.sidebar.title("🎛️ Control Panel")
    
    # Data management section
    with st.sidebar.expander("📊 Data Management", expanded=True):
        if st.button("🔄 Load Data", type="primary", help="Collect market data and generate customer data"):
            load_data()
        
        if st.session_state.data_loaded:
            st.success("✅ Data loaded")
            
            # Data info
            aggregated_data = st.session_state.aggregated_data
            if aggregated_data:
                market_data = aggregated_data.get('market_data', pd.DataFrame())
                customer_data = aggregated_data.get('customer_data', {})
                
                if not market_data.empty:
                    st.write(f"📈 Market records: {len(market_data)}")
                
                if customer_data:
                    customers = customer_data.get('customers', pd.DataFrame())
                    flows = customer_data.get('flows', pd.DataFrame())
                    st.write(f"👥 Customers: {len(customers)}")
                    st.write(f"💸 Flow records: {len(flows)}")
        else:
            st.warning("⚠️ No data loaded")
    
    # Model training section
    with st.sidebar.expander("🤖 Model Training", expanded=st.session_state.data_loaded):
        if st.session_state.data_loaded:
            if st.button("🎯 Train Models", type="secondary", help="Train ML models for deposit prediction"):
                train_models()
            
            if st.session_state.models_trained:
                st.success("✅ Models trained")
                
                models = st.session_state.models.get('individual_models', {})
                best_model_name = models.get('best_model_name', 'N/A')
                st.write(f"🏆 Best model: {best_model_name}")
            else:
                st.info("ℹ️ Models not trained")
        else:
            st.info("ℹ️ Load data first")
    
    # Settings section
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        st.write("**Display Options**")
        
        # Theme selection
        theme = st.selectbox("Color Theme", ["Default", "Dark", "Light"])
        
        # Chart settings
        chart_height = st.slider("Chart Height", 300, 800, 400)
        
        # Update session state with settings
        st.session_state.chart_height = chart_height

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    
    pages = {
        "🏠 Main Dashboard": display_main_dashboard,
        "📋 Data Overview": display_data_overview,
        "🧠 Behavioral Analysis": display_behavioral_analysis
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    
    # Display selected page
    try:
        pages[selected_page]()
    except Exception as e:
        st.error(f"Error displaying page: {e}")
        st.info("Please try refreshing the page or contact support.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;'>
            🏦 Treasury Behavioral Deposit Analytics Dashboard<br>
            Built with Streamlit • Real-time Market Data • Advanced ML Models<br>
            <em>Designed for Commerzbank Treasury Internship</em>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page to restart the application.")