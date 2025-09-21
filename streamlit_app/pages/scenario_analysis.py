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
        st.title("Scenario Analysis")
        st.markdown("Comprehensive stress testing and scenario impact analysis")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load data first to run scenario analysis")
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
        st.subheader("Scenario Configuration")
        
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
        if st.button("Run Scenario Analysis", type="primary"):
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
        scenario_lcr = base_lcr * (1 - deposit_impact * 0.5)  # LCR decreases with outflows
        scenario_flows = base_flows * (1 + rate_impact * 10)  # Flows affected by rates
        
        # Calculate risks
        funding_gap = max(0, (base_deposits - scenario_deposits) - (base_deposits * 0.15))
        stress_ratio = scenario_lcr / base_lcr
        
        results = {
            'base_case': {
                'deposits': base_deposits,
                'lcr_ratio': base_lcr,
                'daily_flows': base_flows
            },
            'scenario_case': {
                'deposits': scenario_deposits,
                'lcr_ratio': scenario_lcr,
                'daily_flows': scenario_flows
            },
            'impacts': {
                'deposit_change': scenario_deposits - base_deposits,
                'deposit_change_pct': (scenario_deposits - base_deposits) / base_deposits * 100,
                'lcr_change': scenario_lcr - base_lcr,
                'flow_change': scenario_flows - base_flows,
                'funding_gap': funding_gap,
                'stress_ratio': stress_ratio
            },
            'risk_assessment': {
                'risk_level': 'High' if stress_ratio < 0.8 else 'Medium' if stress_ratio < 0.9 else 'Low',
                'lcr_breach': scenario_lcr < 1.0,
                'critical_metrics': []
            }
        }
        
        # Add critical metrics
        if scenario_lcr < 1.0:
            results['risk_assessment']['critical_metrics'].append('LCR below regulatory minimum')
        if deposit_impact > 0.2:
            results['risk_assessment']['critical_metrics'].append('High deposit outflow')
        if funding_gap > base_deposits * 0.1:
            results['risk_assessment']['critical_metrics'].append('Significant funding gap')
        
        return results
        
    except Exception as e:
        st.error(f"Error calculating scenario impacts: {e}")
        return {}

def display_scenario_results(results: dict, scenario_type: str, severity_level: str):
    """Display scenario analysis results"""
    try:
        if not results:
            st.error("No scenario results to display")
            return
        
        st.success(f"Scenario analysis completed: {scenario_type} - {severity_level}")
        
        # Impact summary
        st.subheader("Impact Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        impacts = results.get('impacts', {})
        
        with col1:
            deposit_change = impacts.get('deposit_change_pct', 0)
            st.metric(
                "Deposit Change", 
                f"{deposit_change:+.1f}%",
                help="Change in total deposits"
            )
        
        with col2:
            lcr_change = impacts.get('lcr_change', 0)
            st.metric(
                "LCR Change",
                f"{lcr_change:+.3f}",
                help="Change in Liquidity Coverage Ratio"
            )
        
        with col3:
            funding_gap = impacts.get('funding_gap', 0)
            st.metric(
                "Funding Gap",
                f"${funding_gap/1e6:.1f}M",
                help="Additional funding needed"
            )
        
        with col4:
            risk_level = results.get('risk_assessment', {}).get('risk_level', 'Unknown')
            st.metric(
                "Risk Level",
                risk_level,
                help="Overall risk assessment"
            )
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Waterfall chart
            st.subheader("Impact Breakdown")
            
            base_deposits = results.get('base_case', {}).get('deposits', 0)
            scenario_deposits = results.get('scenario_case', {}).get('deposits', 0)
            
            if base_deposits > 0:
                create_scenario_waterfall_chart(
                    base_deposits, 
                    {"Scenario Impact": scenario_deposits}
                )
        
        with col2:
            # Risk assessment
            st.subheader("Risk Assessment")
            
            risk_assessment = results.get('risk_assessment', {})
            
            # Risk level indicator
            risk_level = risk_assessment.get('risk_level', 'Unknown')
            if risk_level == 'High':
                st.error(f"Risk Level: {risk_level}")
            elif risk_level == 'Medium':
                st.warning(f"Risk Level: {risk_level}")
            else:
                st.success(f"Risk Level: {risk_level}")
            
            # Critical metrics
            critical_metrics = risk_assessment.get('critical_metrics', [])
            if critical_metrics:
                st.write("**Critical Issues:**")
                for metric in critical_metrics:
                    st.write(f"• {metric}")
            else:
                st.write("**No critical issues identified**")
        
        # Store results for comparison
        if 'scenario_results' not in st.session_state:
            st.session_state.scenario_results = []
        
        st.session_state.scenario_results.append({
            'timestamp': datetime.now(),
            'type': scenario_type,
            'severity': severity_level,
            'results': results
        })
        
    except Exception as e:
        st.error(f"Error displaying scenario results: {e}")

def display_stress_test_scenarios():
    """Display predefined stress test scenarios"""
    try:
        st.subheader("Predefined Stress Tests")
        
        # Define stress test scenarios
        stress_scenarios = {
            "2008 Financial Crisis": {
                "description": "Severe market crisis with high VIX and credit spreads",
                "params": {"vix_shock": 40, "deposit_outflow": 0.25, "credit_spread": 500}
            },
            "COVID-19 Pandemic": {
                "description": "Economic uncertainty with moderate market stress",
                "params": {"vix_shock": 30, "deposit_outflow": 0.15, "economic_shock": "mild_recession"}
            },
            "Interest Rate Surge": {
                "description": "Rapid increase in interest rates",
                "params": {"rate_change": 400, "deposit_outflow": 0.20, "vix_increase": 15}
            },
            "Banking Sector Stress": {
                "description": "Sector-specific stress affecting liquidity",
                "params": {"deposit_outflow": 0.30, "credit_spread": 300, "vix_shock": 25}
            }
        }
        
        col1, col2 = st.columns(2)
        
        for i, (scenario_name, scenario_info) in enumerate(stress_scenarios.items()):
            col = col1 if i % 2 == 0 else col2
            
            with col:
                with st.expander(scenario_name):
                    st.write(scenario_info["description"])
                    st.write("**Parameters:**")
                    for param, value in scenario_info["params"].items():
                        if isinstance(value, float) and param.endswith(('_rate', '_outflow')):
                            st.write(f"• {param}: {value:.1%}")
                        else:
                            st.write(f"• {param}: {value}")
                    
                    if st.button(f"Run {scenario_name}", key=f"stress_{i}"):
                        st.session_state.scenario_params = scenario_info["params"]
                        results = calculate_scenario_impacts(scenario_info["params"], 
                                                           st.session_state.aggregated_data.get('customer_data', {}))
                        display_scenario_results(results, scenario_name, "Historical")
        
    except Exception as e:
        st.error(f"Error displaying stress test scenarios: {e}")

def display_custom_scenario_builder():
    """Display custom scenario builder"""
    try:
        st.subheader("Custom Scenario Builder")
        
        # Advanced scenario options
        with st.expander("Advanced Scenario Options", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Factors**")
                equity_shock = st.slider("Equity Market Decline (%)", 0, 70, 0, 5)
                fx_shock = st.slider("FX Volatility Spike", 0, 200, 0, 10)
                commodity_shock = st.slider("Commodity Price Change (%)", -50, 100, 0, 5)
            
            with col2:
                st.write("**Economic Factors**")
                gdp_impact = st.slider("GDP Impact (%)", -10, 5, 0, 1)
                inflation_shock = st.slider("Inflation Change (pp)", -2, 5, 0, 0.5)
                employment_impact = st.slider("Unemployment Change (pp)", 0, 10, 0, 1)
            
            # Combined scenario parameters
            custom_params = {
                "equity_shock": equity_shock / 100,
                "fx_shock": fx_shock / 100,
                "commodity_shock": commodity_shock / 100,
                "gdp_impact": gdp_impact / 100,
                "inflation_shock": inflation_shock / 100,
                "employment_impact": employment_impact / 100
            }
            
            if st.button("Run Custom Scenario"):
                results = calculate_scenario_impacts(custom_params, 
                                                   st.session_state.aggregated_data.get('customer_data', {}))
                display_scenario_results(results, "Custom", "User Defined")
        
    except Exception as e:
        st.error(f"Error displaying custom scenario builder: {e}")

def display_scenario_comparison():
    """Display comparison of multiple scenarios"""
    try:
        st.subheader("Scenario Comparison")
        
        scenario_results = st.session_state.get('scenario_results', [])
        
        if len(scenario_results) < 2:
            st.info("Run at least 2 scenarios to see comparison")
            return
        
        # Create comparison data
        comparison_data = []
        
        for result in scenario_results[-5:]:  # Last 5 scenarios
            impacts = result['results'].get('impacts', {})
            comparison_data.append({
                'Scenario': f"{result['type']} ({result['severity']})",
                'Deposit Change (%)': impacts.get('deposit_change_pct', 0),
                'LCR Change': impacts.get('lcr_change', 0),
                'Funding Gap ($M)': impacts.get('funding_gap', 0) / 1e6,
                'Risk Level': result['results'].get('risk_assessment', {}).get('risk_level', 'Unknown')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Deposit change comparison
            fig = px.bar(
                comparison_df,
                x='Scenario',
                y='Deposit Change (%)',
                title="Deposit Impact Comparison",
                color='Deposit Change (%)',
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level comparison
            risk_colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
            
            fig = px.scatter(
                comparison_df,
                x='LCR Change',
                y='Funding Gap ($M)',
                size='Deposit Change (%)',
                color='Risk Level',
                color_discrete_map=risk_colors,
                title="Risk vs Impact Analysis",
                hover_data=['Scenario']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying scenario comparison: {e}")

if __name__ == "__main__":
    # Test the scenario analysis page
    st.title("Scenario Analysis Page Test")
    display_scenario_analysis()