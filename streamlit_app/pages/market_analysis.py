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
from streamlit_app.components.charts import create_yield_curve_chart

def display_market_analysis():
    """Market analysis page"""
    try:
        st.title("📊 Market Analysis")
        st.markdown("Comprehensive analysis of market conditions and their impact on deposit behavior")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please load data first to view market analysis")
            return
        
        # Get market data
        aggregated_data = st.session_state.aggregated_data
        market_data = aggregated_data.get('market_data', pd.DataFrame())
        
        if market_data.empty:
            st.error("❌ No market data available for analysis")
            return
        
        # Market overview section
        display_market_overview(market_data)
        
        st.markdown("---")
        
        # Interest rate analysis
        display_interest_rate_analysis(market_data)
        
        st.markdown("---")
        
        # Market volatility analysis
        display_volatility_analysis(market_data)
        
        st.markdown("---")
        
        # Correlation analysis
        display_correlation_analysis(market_data)
        
    except Exception as e:
        st.error(f"Error displaying market analysis: {e}")

def display_market_overview(market_data: pd.DataFrame):
    """Display market overview section"""
    try:
        st.subheader("🌍 Market Overview")
        
        # Current market conditions
        latest_data = market_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fed_rate = latest_data.get('FED_FUNDS_RATE', 0)
            rate_change = market_data['FED_FUNDS_RATE'].diff().iloc[-1] if 'FED_FUNDS_RATE' in market_data.columns else 0
            st.metric(
                "Fed Funds Rate",
                f"{fed_rate:.2f}%",
                f"{rate_change:+.2f}%"
            )
        
        with col2:
            treasury_10y = latest_data.get('TREASURY_10Y', 0)
            treasury_change = market_data['TREASURY_10Y'].diff().iloc[-1] if 'TREASURY_10Y' in market_data.columns else 0
            st.metric(
                "10Y Treasury",
                f"{treasury_10y:.2f}%",
                f"{treasury_change:+.2f}%"
            )
        
        with col3:
            vix = latest_data.get('VIX', 0)
            vix_change = market_data['VIX'].diff().iloc[-1] if 'VIX' in market_data.columns else 0
            st.metric(
                "VIX Level",
                f"{vix:.1f}",
                f"{vix_change:+.1f}"
            )
        
        with col4:
            if all(col in market_data.columns for col in ['TREASURY_10Y', 'TREASURY_2Y']):
                yield_spread = latest_data['TREASURY_10Y'] - latest_data['TREASURY_2Y']
                spread_change = (market_data['TREASURY_10Y'] - market_data['TREASURY_2Y']).diff().iloc[-1]
                st.metric(
                    "10Y-2Y Spread",
                    f"{yield_spread:.1f}bps",
                    f"{spread_change:+.1f}bps"
                )
        
        # Market regime analysis
        st.subheader("📈 Market Regime Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interest rate regime
            fed_rate = latest_data.get('FED_FUNDS_RATE', 2.0)
            
            if fed_rate <= 0.25:
                rate_regime = "Zero Interest Rate Policy"
                regime_color = "🔵"
            elif fed_rate <= 2.0:
                rate_regime = "Low Rate Environment"
                regime_color = "🟢"
            elif fed_rate <= 5.0:
                rate_regime = "Normal Rate Environment"
                regime_color = "🟡"
            else:
                rate_regime = "High Rate Environment"
                regime_color = "🔴"
            
            st.info(f"{regime_color} **Interest Rate Regime**: {rate_regime}")
        
        with col2:
            # Volatility regime
            vix = latest_data.get('VIX', 20)
            
            if vix < 15:
                vol_regime = "Low Volatility"
                vol_color = "🟢"
            elif vix < 25:
                vol_regime = "Normal Volatility"
                vol_color = "🟡"
            elif vix < 35:
                vol_regime = "High Volatility"
                vol_color = "🟠"
            else:
                vol_regime = "Extreme Volatility"
                vol_color = "🔴"
            
            st.info(f"{vol_color} **Volatility Regime**: {vol_regime}")
    
    except Exception as e:
        st.error(f"Error displaying market overview: {e}")

def display_interest_rate_analysis(market_data: pd.DataFrame):
    """Display interest rate analysis section"""
    try:
        st.subheader("📈 Interest Rate Analysis")
        
        # Rate time series
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Interest Rate Evolution**")
            
            # Create rate time series chart
            rate_columns = [col for col in market_data.columns if 'RATE' in col or 'TREASURY' in col]
            
            if rate_columns:
                fig = go.Figure()
                
                colors = VIZ_CONFIG['COLORS']['GRADIENT']
                
                for i, col in enumerate(rate_columns[:4]):  # Limit to 4 series
                    if col in market_data.columns:
                        color = colors[i % len(colors)]
                        fig.add_trace(go.Scatter(
                            x=market_data.index,
                            y=market_data[col],
                            mode='lines',
                            name=col.replace('_', ' ').title(),
                            line=dict(color=color, width=2)
                        ))
                
                fig.update_layout(
                    title="Interest Rate Time Series",
                    xaxis_title="Date",
                    yaxis_title="Rate (%)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No interest rate data available")
        
        with col2:
            st.write("**Current Yield Curve**")
            create_yield_curve_chart(market_data)
        
        # Yield curve analysis
        if all(col in market_data.columns for col in ['TREASURY_10Y', 'TREASURY_2Y', 'TREASURY_3M']):
            st.subheader("📊 Yield Curve Dynamics")
            
            # Calculate spreads
            market_data_spreads = market_data.copy()
            market_data_spreads['10Y_2Y_Spread'] = market_data['TREASURY_10Y'] - market_data['TREASURY_2Y']
            market_data_spreads['2Y_3M_Spread'] = market_data['TREASURY_2Y'] - market_data['TREASURY_3M']
            
            # Spread chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('10Y-2Y Spread', '2Y-3M Spread'),
                shared_xaxes=True
            )
            
            fig.add_trace(
                go.Scatter(
                    x=market_data_spreads.index,
                    y=market_data_spreads['10Y_2Y_Spread'],
                    mode='lines',
                    name='10Y-2Y',
                    line=dict(color=VIZ_CONFIG['COLORS']['PRIMARY'])
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=market_data_spreads.index,
                    y=market_data_spreads['2Y_3M_Spread'],
                    mode='lines',
                    name='2Y-3M',
                    line=dict(color=VIZ_CONFIG['COLORS']['SECONDARY'])
                ),
                row=2, col=1
            )
            
            # Add zero lines
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
            
            fig.update_layout(height=500, title_text="Yield Curve Spreads")
            fig.update_yaxes(title_text="Spread (bps)", row=1, col=1)
            fig.update_yaxes(title_text="Spread (bps)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Spread statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**10Y-2Y Spread Statistics**")
                spread_10y2y = market_data_spreads['10Y_2Y_Spread']
                
                stats_df = pd.DataFrame({
                    'Metric': ['Current', 'Average', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{spread_10y2y.iloc[-1]:.2f}",
                        f"{spread_10y2y.mean():.2f}",
                        f"{spread_10y2y.std():.2f}",
                        f"{spread_10y2y.min():.2f}",
                        f"{spread_10y2y.max():.2f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.write("**2Y-3M Spread Statistics**")
                spread_2y3m = market_data_spreads['2Y_3M_Spread']
                
                stats_df = pd.DataFrame({
                    'Metric': ['Current', 'Average', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{spread_2y3m.iloc[-1]:.2f}",
                        f"{spread_2y3m.mean():.2f}",
                        f"{spread_2y3m.std():.2f}",
                        f"{spread_2y3m.min():.2f}",
                        f"{spread_2y3m.max():.2f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying interest rate analysis: {e}")

def display_volatility_analysis(market_data: pd.DataFrame):
    """Display market volatility analysis section"""
    try:
        st.subheader("⚡ Market Volatility Analysis")
        
        if 'VIX' not in market_data.columns:
            st.info("VIX data not available for volatility analysis")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # VIX time series with regime bands
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=market_data.index,
                y=market_data['VIX'],
                mode='lines',
                name='VIX',
                line=dict(color=VIZ_CONFIG['COLORS']['DANGER'], width=2)
            ))
            
            # Add regime bands
            fig.add_hrect(y0=0, y1=15, fillcolor="green", opacity=0.1, 
                         annotation_text="Low Volatility", annotation_position="top left")
            fig.add_hrect(y0=15, y1=25, fillcolor="yellow", opacity=0.1,
                         annotation_text="Normal Volatility", annotation_position="top left")
            fig.add_hrect(y0=25, y1=35, fillcolor="orange", opacity=0.1,
                         annotation_text="High Volatility", annotation_position="top left")
            fig.add_hrect(y0=35, y1=100, fillcolor="red", opacity=0.1,
                         annotation_text="Extreme Volatility", annotation_position="top left")
            
            fig.update_layout(
                title="VIX Evolution with Regime Bands",
                xaxis_title="Date",
                yaxis_title="VIX Level",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # VIX distribution
            st.write("**VIX Distribution**")
            
            fig = px.histogram(
                x=market_data['VIX'],
                nbins=20,
                title="VIX Distribution",
                color_discrete_sequence=[VIZ_CONFIG['COLORS']['SECONDARY']]
            )
            
            fig.update_layout(
                xaxis_title="VIX Level",
                yaxis_title="Frequency",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Volatility statistics and regimes
        st.subheader("📊 Volatility Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_vix = market_data['VIX'].iloc[-1]
            avg_vix = market_data['VIX'].mean()
            st.metric("Current VIX", f"{current_vix:.1f}", f"{current_vix - avg_vix:+.1f}")
        
        with col2:
            vix_volatility = market_data['VIX'].rolling(21).std().iloc[-1]
            st.metric("VIX Volatility (21d)", f"{vix_volatility:.1f}")
        
        with col3:
            vix_percentile = (market_data['VIX'] <= current_vix).mean() * 100
            st.metric("Current Percentile", f"{vix_percentile:.0f}%")
        
        # Regime analysis
        vix_regimes = {
            'Low Volatility (VIX < 15)': (market_data['VIX'] < 15).sum(),
            'Normal Volatility (15 ≤ VIX < 25)': ((market_data['VIX'] >= 15) & (market_data['VIX'] < 25)).sum(),
            'High Volatility (25 ≤ VIX < 35)': ((market_data['VIX'] >= 25) & (market_data['VIX'] < 35)).sum(),
            'Extreme Volatility (VIX ≥ 35)': (market_data['VIX'] >= 35).sum()
        }
        
        regime_df = pd.DataFrame({
            'Regime': list(vix_regimes.keys()),
            'Days': list(vix_regimes.values()),
            'Percentage': [v/len(market_data)*100 for v in vix_regimes.values()]
        })
        
        st.write("**Volatility Regime Distribution**")
        st.dataframe(regime_df, hide_index=True, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying volatility analysis: {e}")

def display_correlation_analysis(market_data: pd.DataFrame):
    """Display correlation analysis section"""
    try:
        st.subheader("🔗 Market Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_columns = market_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to key market variables
        key_variables = []
        for col in numeric_columns:
            if any(keyword in col.upper() for keyword in ['RATE', 'TREASURY', 'VIX', 'SPREAD', 'GDP', 'INFLATION']):
                key_variables.append(col)
        
        if len(key_variables) < 2:
            st.info("Insufficient market variables for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = market_data[key_variables].corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Correlation heatmap
            fig = px.imshow(
                corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu',
                aspect='auto',
                title="Market Variables Correlation Matrix"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Key Correlations**")
            
            # Find strongest correlations (excluding diagonal)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        corr_pairs.append({
                            'Pair': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                            'Correlation': corr_val
                        })
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x['Correlation']), reverse=True)
            
            # Display top correlations
            for pair in corr_pairs[:10]:  # Top 10
                corr_val = pair['Correlation']
                if abs(corr_val) > 0.3:  # Only show meaningful correlations
                    color = "🔴" if corr_val < -0.7 else "🟠" if corr_val < -0.3 else "🟡" if abs(corr_val) < 0.7 else "🟢"
                    st.write(f"{color} {corr_val:+.2f}: {pair['Pair'][:30]}...")
        
        # Time-varying correlations
        if len(key_variables) >= 2:
            st.subheader("📈 Rolling Correlations")
            
            # Select two key variables for rolling correlation
            var1 = 'FED_FUNDS_RATE' if 'FED_FUNDS_RATE' in key_variables else key_variables[0]
            var2 = 'VIX' if 'VIX' in key_variables else key_variables[1] if len(key_variables) > 1 else key_variables[0]
            
            if var1 != var2:
                # Calculate 30-day rolling correlation
                rolling_corr = market_data[var1].rolling(30).corr(market_data[var2])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=market_data.index,
                    y=rolling_corr,
                    mode='lines',
                    name=f'{var1} vs {var2}',
                    line=dict(color=VIZ_CONFIG['COLORS']['PRIMARY'], width=2)
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.5)
                fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5)
                
                fig.update_layout(
                    title=f"30-Day Rolling Correlation: {var1} vs {var2}",
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying correlation analysis: {e}")

if __name__ == "__main__":
    # Test the market analysis page
    st.title("Market Analysis Page Test")
    
    # Mock session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = True
        st.session_state.aggregated_data = {
            'market_data': pd.DataFrame({
                'FED_FUNDS_RATE': np.random.randn(100) + 2,
                'TREASURY_10Y': np.random.randn(100) + 3,
                'VIX': np.random.randn(100) * 5 + 20
            }, index=pd.date_range('2023-01-01', periods=100))
        }
    
    display_market_analysis()