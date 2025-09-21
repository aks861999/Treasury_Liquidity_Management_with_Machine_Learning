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

from config import VIZ_CONFIG, RISK_CONFIG
from streamlit_app.components.charts import create_scenario_waterfall_chart, create_risk_gauge_chart

def display_scenario_analysis():
    """Scenario analysis page"""
    try:
        st.title("🎯 Scenario Analysis")
        st.markdown("Comprehensive stress testing and scenario impact analysis")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please load data first to run scenario analysis")
            return
        
        # Scenario configuration
        display_scenario_configuration()
        
        st.markdown("---")
        
        # Stress test scenarios
        display_stress_test_scenarios()
        
        st.markdown("---")
        
        # Custom scenario builder
        display_custom_scenario_builder()
        
        st.markdown("---")
        
        # Scenario comparison
        display_scenario_comparison()
        
    except Exception as e:
        st.error(f"Error displaying scenario analysis: {e}")

def display_scenario_configuration():
    """Display scenario configuration section"""
    try:
        st.subheader("⚙️ Scenario Configuration")
        
        # Quick scenario selector
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scenario_type = st.selectbox(
                "Scenario Type",
                ["Interest Rate Shock", "Market Crisis", "Economic Recession", "Custom"]
            )
        
        with col2:
            severity_level = st.selectbox(
                "Severity Level",
                ["Mild", "Moderate", "Severe"]
            )
        
        with col3:
            time_horizon = st.selectbox(
                "Time Horizon",
                ["1 Month", "3 Months", "6 Months", "1 Year"]
            )
        
        # Scenario parameters based on selection
        if scenario_type == "Interest Rate Shock":
            display_rate_shock_config(severity_level)
        elif scenario_type == "Market Crisis":
            display_market_crisis_config(severity_level)
        elif scenario_type == "Economic Recession":
            display_recession_config(severity_level)
        else:
            display_custom_config()
        
        # Run scenario button
        if st.button("🔄 Run Scenario Analysis", type="primary"):
            run_scenario_analysis(scenario_type, severity_level, time_horizon)
    
    except Exception as e:
        st.error(f"Error displaying scenario configuration: {e}")

def display_rate_shock_config(severity_level: str):
    """Display interest rate shock configuration"""
    try:
        st.write("**Interest Rate Shock Parameters**")
        
        # Predefined shocks based on severity
        shock_params = {
            "Mild": {"rate_change": 100, "deposit_outflow": 5, "vix_increase": 5},
            "Moderate": {"rate_change": 250, "deposit_outflow": 15, "vix_increase": 15},
            "Severe": {"rate_change": 500, "deposit_outflow": 30, "vix_increase": 25}
        }
        
        params = shock_params.get(severity_level, shock_params["Moderate"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rate_change = st.number_input(
                "Interest Rate Change (bps)",
                value=params["rate_change"],
                min_value=-1000,
                max_value=1000,
                step=25
            )
        
        with col2:
            deposit_outflow = st.number_input(
                "Deposit Outflow (%)",
                value=params["deposit_outflow"],
                min_value=0,
                max_value=50,
                step=1
            )
        
        with col3:
            vix_increase = st.number_input(
                "VIX Increase",
                value=params["vix_increase"],
                min_value=0,
                max_value=50,
                step=1
            )
        
        # Store parameters in session state
        st.session_state.scenario_params = {
            "rate_change": rate_change,
            "deposit_outflow": deposit_outflow / 100,
            "vix_increase": vix_increase
        }
    
    except Exception as e:
        st.error(f"Error displaying rate shock config: {e}")

def display_market_crisis_config(severity_level: str):
    """Display market crisis configuration"""
    try:
        st.write("**Market Crisis Parameters**")
        
        crisis_params = {
            "Mild": {"vix_level": 35, "credit_spread": 200, "deposit_outflow": 10},
            "Moderate": {"vix_level": 45, "credit_spread": 400, "deposit_outflow": 20},
            "Severe": {"vix_level": 60, "credit_spread": 600, "deposit_outflow": 35}
        }
        
        params = crisis_params.get(severity_level, crisis_params["Moderate"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vix_level = st.number_input(
                "VIX Peak Level",
                value=params["vix_level"],
                min_value=20,
                max_value=80,
                step=5
            )
        
        with col2:
            credit_spread = st.number_input(
                "Credit Spread Widening (bps)",
                value=params["credit_spread"],
                min_value=0,
                max_value=1000,
                step=50
            )
        
        with col3:
            deposit_outflow = st.number_input(
                "Deposit Outflow (%)",
                value=params["deposit_outflow"],
                min_value=0,
                max_value=50,
                step=1
            )
        
        st.session_state.scenario_params = {
            "vix_shock": vix_level - 20,  # Assuming base VIX of 20
            "credit_spread": credit_spread,
            "deposit_outflow": deposit_outflow / 100
        }
    
    except Exception as e:
        st.error(f"Error displaying market crisis config: {e}")

def display_recession_config(severity_level: str):
    """Display economic recession configuration"""
    try:
        st.write("**Economic Recession Parameters**")
        
        recession_params = {
            "Mild": {"gdp_decline": -2, "unemployment_rise": 2, "deposit_outflow": 8},
            "Moderate": {"gdp_decline": -5, "unemployment_rise": 4, "deposit_outflow": 18},
            "Severe": {"gdp_decline": -8, "unemployment_rise": 6, "deposit_outflow": 30}
        }
        
        params = recession_params.get(severity_level, recession_params["Moderate"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gdp_decline = st.number_input(
                "GDP Decline (%)",
                value=params["gdp_decline"],
                min_value=-15,
                max_value=0,
                step=1
            )
        
        with col2:
            unemployment_rise = st.number_input(
                "Unemployment Rise (%)",
                value=params["unemployment_rise"],
                min_value=0,
                max_value=10,
                step=1
            )
        
        with col3:
            deposit_outflow = st.number_input(
                "Deposit Outflow (%)",
                value=params["deposit_outflow"],
                min_value=0,
                max_value=50,
                step=1
            )
        
        st.session_state.scenario_params = {
            "economic_shock": "recession",
            "gdp_impact": gdp_decline / 100,
            "unemployment_impact": unemployment_rise / 100,
            "deposit_outflow": deposit_outflow / 100
        }
    
    except Exception as e:
        st.error(f"Error displaying recession config: {e}")

def display_custom_config():
    """Display custom scenario configuration"""
    try:
        st.write("**Custom Scenario Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Market Conditions**")
            rate_change = st.slider("Interest Rate Change (bps)", -500, 500, 0, 25)
            vix_change = st.slider("VIX Change", -10, 50, 0, 5)
            fx_volatility = st.slider("FX Volatility Increase (%)", 0, 100, 0, 5)
        
        with col2:
            st.write("**Behavioral Impact**")
            deposit_outflow = st.slider("Deposit Outflow (%)", 0, 50, 0, 1)
            customer_sensitivity = st.slider("Customer Sensitivity Multiplier", 0.5, 3.0, 1.0, 0.1)
            segment_impact = st.selectbox("Most Affected Segment", 
                                        ["All Equally", "Retail", "SME", "Corporate"])
        
        st.session_state.scenario_params = {
            "rate_change": rate_change,
            "vix_change": vix_change,
            "fx_volatility": fx_volatility / 100,
            "deposit_outflow": deposit_outflow / 100,
            "sensitivity_multiplier": customer_sensitivity,
            "segment_focus": segment_impact
        }
    
    except Exception as e:
        st.error(f"Error displaying custom config: {e}")

def run_scenario_analysis(scenario_type: str, severity_level: str, time_horizon: str):
    """Run the scenario analysis"""
    try:
        with st.spinner("Running scenario analysis..."):
            # Get scenario parameters
            params = st.session_state.get('scenario_params', {})
            
            # Get base data
            aggregated_data = st.session_state.aggregated_data
            customer_data = aggregated_data.get('customer_data', {})
            
            # Calculate scenario impacts
            results = calculate_scenario_impacts(params, customer_data)
            
            # Display results
            display_scenario_results(results, scenario_type, severity_level)
    
    except Exception as e:
        st.error(f"Error running scenario analysis: {e}")

def calculate_scenario_impacts(params: dict, customer_data: dict) -> dict:
    """Calculate impacts of scenario on key metrics"""
    try:
        # Base case values
        customers = customer_data.get('customers', pd.DataFrame())
        flows = customer_data.get('flows', pd.DataFrame())
        
        base_deposits = customers['balance_avg'].sum() if not customers.empty else 1e9
        base_lcr = 1.15  # Mock base LCR
        base_flows = flows['deposit_flow'].sum() if not flows.empty else 0
        
        # Calculate impacts
        deposit_impact = params.get('deposit_outflow', 0)
        rate_impact = params.get('rate_change', 0) / 10000  # Convert bps to decimal
        vix_impact = params.get('vix_change', 0) / 100
        
        # Scenario calculations
        scenario_deposits = base_deposits * (1 - deposit_impact)