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
from streamlit_app.components.charts import create_behavioral_analysis_chart

def display_behavioral_modeling():
    """Behavioral modeling analysis page"""
    try:
        st.title("🧠 Behavioral Modeling")
        st.markdown("Deep dive into customer behavior patterns and predictive modeling")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please load data first to view behavioral analysis")
            return
        
        # Get data
        aggregated_data = st.session_state.aggregated_data
        customer_data = aggregated_data.get('customer_data', {})
        
        customers = customer_data.get('customers', pd.DataFrame())
        flows = customer_data.get('flows', pd.DataFrame())
        
        if customers.empty:
            st.error("❌ No customer data available for behavioral analysis")
            return
        
        # Behavioral overview
        display_behavioral_overview(customers, flows)
        
        st.markdown("---")
        
        # Customer segmentation analysis
        display_segmentation_analysis(customers, flows)
        
        st.markdown("---")
        
        # Rate sensitivity analysis
        display_rate_sensitivity_analysis(customers, flows)
        
        st.markdown("---")
        
        # Behavioral predictions
        display_behavioral_predictions()
        
    except Exception as e:
        st.error(f"Error displaying behavioral modeling: {e}")

def display_behavioral_overview(customers: pd.DataFrame, flows: pd.DataFrame):
    """Display behavioral overview section"""
    try:
        st.subheader("📊 Behavioral Overview")
        
        # Key behavioral metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'rate_sensitivity' in customers.columns:
                avg_rate_sensitivity = customers['rate_sensitivity'].mean()
                st.metric("Avg Rate Sensitivity", f"{avg_rate_sensitivity:.2f}")
            else:
                st.metric("Avg Rate Sensitivity", "N/A")
        
        with col2:
            if 'loyalty_score' in customers.columns:
                avg_loyalty = customers['loyalty_score'].mean()
                st.metric("Avg Loyalty Score", f"{avg_loyalty:.2f}")
            else:
                st.metric("Avg Loyalty Score", "N/A")
        
        with col3:
            if 'volatility_factor' in customers.columns:
                avg_volatility = customers['volatility_factor'].mean()
                st.metric("Avg Volatility Factor", f"{avg_volatility:.2f}")
            else:
                st.metric("Avg Volatility Factor", "N/A")
        
        with col4:
            if not flows.empty and 'deposit_flow' in flows.columns:
                flow_stability = 1 - (flows['deposit_flow'].std() / flows['deposit_flow'].abs().mean())
                flow_stability = max(0, min(1, flow_stability))
                st.metric("Flow Stability Index", f"{flow_stability:.2f}")
            else:
                st.metric("Flow Stability Index", "N/A")
        
        # Behavioral distribution charts
        st.subheader("📈 Behavioral Characteristics Distribution")
        
        behavioral_cols = ['rate_sensitivity', 'loyalty_score', 'volatility_factor']
        available_cols = [col for col in behavioral_cols if col in customers.columns]
        
        if available_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution plots
                for i, col in enumerate(available_cols[:2]):
                    fig = px.histogram(
                        customers, 
                        x=col,
                        title=f"Distribution of {col.replace('_', ' ').title()}",
                        nbins=20,
                        color_discrete_sequence=[VIZ_CONFIG['COLORS']['GRADIENT'][i % len(VIZ_CONFIG['COLORS']['GRADIENT'])]]
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plots by segment
                if 'segment' in customers.columns:
                    for i, col in enumerate(available_cols[:2]):
                        fig = px.box(
                            customers,
                            x='segment',
                            y=col,
                            title=f"{col.replace('_', ' ').title()} by Segment",
                            color='segment',
                            color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT']
                        )
                        fig.update_layout(height=300, showlegend=False)
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Behavioral characteristics data not available")
    
    except Exception as e:
        st.error(f"Error displaying behavioral overview: {e}")

def display_segmentation_analysis(customers: pd.DataFrame, flows: pd.DataFrame):
    """Display customer segmentation analysis"""
    try:
        st.subheader("🎯 Customer Segmentation Analysis")
        
        if 'segment' not in customers.columns:
            st.info("Customer segment data not available")
            return
        
        # Segment overview
        segment_stats = customers.groupby('segment').agg({
            'customer_id': 'count',
            'balance_avg': ['sum', 'mean', 'std'],
            'rate_sensitivity': 'mean',
            'loyalty_score': 'mean',
            'volatility_factor': 'mean'
        }).round(3)
        
        # Flatten column names
        segment_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in segment_stats.columns]
        segment_stats = segment_stats.rename(columns={'count_customer_id': 'customer_count'})
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Segment Statistics**")
            st.dataframe(segment_stats, use_container_width=True)
        
        with col2:
            # Segment distribution pie chart
            segment_counts = customers['segment'].value_counts()
            
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution by Segment",
                color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced segmentation analysis
        st.subheader("🔍 Advanced Segment Analysis")
        
        # Multi-dimensional analysis
        if all(col in customers.columns for col in ['balance_avg', 'rate_sensitivity', 'loyalty_score']):
            
            # 3D scatter plot
            fig = px.scatter_3d(
                customers,
                x='balance_avg',
                y='rate_sensitivity', 
                z='loyalty_score',
                color='segment',
                size='volatility_factor' if 'volatility_factor' in customers.columns else None,
                title="3D Customer Segmentation Analysis",
                color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'],
                labels={
                    'balance_avg': 'Average Balance ($)',
                    'rate_sensitivity': 'Rate Sensitivity',
                    'loyalty_score': 'Loyalty Score'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Segment flow analysis
            if not flows.empty and 'customer_id' in flows.columns and 'deposit_flow' in flows.columns:
                st.subheader("💸 Flow Behavior by Segment")
                
                # Merge customer segments with flows
                flows_with_segments = flows.merge(
                    customers[['customer_id', 'segment']], 
                    on='customer_id', 
                    how='left'
                )
                
                # Daily segment flows
                if 'date' in flows_with_segments.columns:
                    daily_segment_flows = flows_with_segments.groupby(['date', 'segment'])['deposit_flow'].sum().reset_index()
                    
                    fig = px.line(
                        daily_segment_flows.tail(200),  # Last 200 data points
                        x='date',
                        y='deposit_flow',
                        color='segment',
                        title="Daily Deposit Flows by Segment",
                        color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Flow statistics by segment
                segment_flow_stats = flows_with_segments.groupby('segment')['deposit_flow'].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).round(2)
                
                st.write("**Flow Statistics by Segment**")
                st.dataframe(segment_flow_stats, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying segmentation analysis: {e}")

def display_rate_sensitivity_analysis(customers: pd.DataFrame, flows: pd.DataFrame):
    """Display rate sensitivity analysis"""
    try:
        st.subheader("📈 Rate Sensitivity Analysis")
        
        if 'rate_sensitivity' not in customers.columns:
            st.info("Rate sensitivity data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rate sensitivity distribution
            fig = px.histogram(
                customers,
                x='rate_sensitivity',
                color='segment' if 'segment' in customers.columns else None,
                title="Rate Sensitivity Distribution",
                nbins=20,
                color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rate sensitivity vs balance
            if 'balance_avg' in customers.columns:
                fig = px.scatter(
                    customers,
                    x='rate_sensitivity',
                    y='balance_avg',
                    color='segment' if 'segment' in customers.columns else None,
                    title="Rate Sensitivity vs Balance",
                    color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'],
                    log_y=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity impact simulation
        st.subheader("🎮 Rate Sensitivity Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Simulation Parameters**")
            rate_change = st.slider("Interest Rate Change (bps)", -500, 500, 0, 25)
            time_horizon = st.selectbox("Time Horizon", ["1 Week", "1 Month", "3 Months", "6 Months"])
        
        with col2:
            if rate_change != 0:
                # Calculate impact
                rate_change_decimal = rate_change / 10000  # Convert bps to decimal
                
                # Estimate deposit flow impact based on sensitivity
                customers_sim = customers.copy()
                if 'balance_avg' in customers_sim.columns:
                    customers_sim['flow_impact'] = (
                        customers_sim['rate_sensitivity'] * 
                        customers_sim['balance_avg'] * 
                        rate_change_decimal * 
                        0.1  # Impact factor
                    )
                    
                    # Aggregate by segment
                    if 'segment' in customers_sim.columns:
                        segment_impact = customers_sim.groupby('segment')['flow_impact'].sum()
                        
                        fig = px.bar(
                            x=segment_impact.index,
                            y=segment_impact.values,
                            title=f"Estimated Flow Impact ({rate_change:+d} bps)",
                            color_discrete_sequence=[VIZ_CONFIG['COLORS']['PRIMARY']]
                        )
                        fig.update_layout(height=300)
                        fig.update_yaxes(title_text="Flow Impact ($)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary statistics
                        total_impact = segment_impact.sum()
                        st.metric("Total Estimated Impact", f"${total_impact:,.0f}")
            else:
                st.info("Adjust the rate change slider to see impact simulation")
    
    except Exception as e:
        st.error(f"Error displaying rate sensitivity analysis: {e}")

def display_behavioral_predictions():
    """Display behavioral predictions section"""
    try:
        st.subheader("🔮 Behavioral Predictions")
        
        if not st.session_state.get('models_trained', False):
            st.info("Train models to see behavioral predictions")
            
            if st.button("🚀 Train Models Now"):
                st.info("Model training would be initiated here...")
            return
        
        # Get model results
        models = st.session_state.models
        individual_models = models.get('individual_models', {})
        
        # Feature importance from behavioral perspective
        if 'best_model_info' in individual_models:
            best_model_info = individual_models['best_model_info']
            feature_importance = best_model_info.get('feature_importance', pd.DataFrame())
            
            if not feature_importance.empty:
                # Filter behavioral features
                behavioral_features = feature_importance[
                    feature_importance['feature'].str.contains(
                        'rate_sensitivity|loyalty|volatility|behavioral|segment',
                        case=False, na=False
                    )
                ]
                
                if not behavioral_features.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top Behavioral Features**")
                        
                        fig = px.bar(
                            behavioral_features.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Behavioral Feature Importance",
                            color_discrete_sequence=[VIZ_CONFIG['COLORS']['SUCCESS']]
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Behavioral Insights**")
                        
                        # Generate insights based on top features
                        top_behavioral = behavioral_features.head(5)
                        
                        for _, feature in top_behavioral.iterrows():
                            importance_pct = feature['importance'] * 100
                            feature_name = feature['feature']
                            
                            if 'rate_sensitivity' in feature_name.lower():
                                insight = "Rate sensitivity is a key driver of deposit behavior"
                            elif 'loyalty' in feature_name.lower():
                                insight = "Customer loyalty significantly impacts deposit stability"
                            elif 'volatility' in feature_name.lower():
                                insight = "Customer volatility factor affects flow predictability"
                            elif 'segment' in feature_name.lower():
                                insight = "Customer segment is important for behavioral prediction"
                            else:
                                insight = "This behavioral factor influences deposit flows"
                            
                            st.write(f"**{importance_pct:.1f}%**: {insight}")
        
        # Behavioral model performance
        performance_data = individual_models.get('performance_comparison', pd.DataFrame())
        
        if not performance_data.empty:
            st.subheader("🎯 Model Performance on Behavioral Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model comparison
                fig = px.bar(
                    performance_data.head(3),
                    x='model_name',
                    y='r2_score',
                    title="Model R² Comparison",
                    color='r2_score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Best model metrics
                best_model = performance_data.iloc[0]
                st.write("**Best Model Performance**")
                st.metric("Model", best_model['model_name'])
                st.metric("R² Score", f"{best_model['r2_score']:.4f}")
                st.metric("RMSE", f"{best_model['rmse']:.4f}")
                st.metric("MAE", f"{best_model['mae']:.4f}")
        
        # Prediction confidence intervals
        st.subheader("📊 Prediction Confidence Analysis")
        
        # Mock prediction confidence data
        confidence_data = pd.DataFrame({
            'Segment': ['RETAIL_SMALL', 'RETAIL_MEDIUM', 'RETAIL_LARGE', 'SME', 'CORPORATE'],
            'Prediction_Accuracy': [0.82, 0.85, 0.88, 0.79, 0.91],
            'Confidence_Interval': [0.15, 0.12, 0.10, 0.18, 0.08]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=confidence_data['Segment'],
            y=confidence_data['Prediction_Accuracy'],
            error_y=dict(type='data', array=confidence_data['Confidence_Interval']),
            name='Prediction Accuracy',
            marker_color=VIZ_CONFIG['COLORS']['PRIMARY']
        ))
        
        fig.update_layout(
            title="Prediction Accuracy by Customer Segment",
            xaxis_title="Customer Segment", 
            yaxis_title="Accuracy",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying behavioral predictions: {e}")

if __name__ == "__main__":
    # Test the behavioral modeling page
    st.title("Behavioral Modeling Page Test")
    display_behavioral_modeling()