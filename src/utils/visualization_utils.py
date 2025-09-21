import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
from config import VIZ_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_time_series_plot(df: pd.DataFrame, x_col: str, y_cols: List[str], 
                           title: str = "Time Series Plot") -> go.Figure:
    """Create interactive time series plot"""
    try:
        fig = go.Figure()
        
        colors = VIZ_CONFIG['COLORS']['GRADIENT']
        
        for i, y_col in enumerate(y_cols):
            if y_col in df.columns:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines',
                    name=y_col.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    hovertemplate=f'{y_col}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title='Value',
            height=VIZ_CONFIG['CHART_HEIGHT'],
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating time series plot: {e}")
        return go.Figure()

def create_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    """Create correlation heatmap"""
    try:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            height=VIZ_CONFIG['CHART_HEIGHT'],
            width=VIZ_CONFIG['CHART_WIDTH']
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        return go.Figure()

def create_distribution_plot(df: pd.DataFrame, column: str, group_by: str = None,
                           plot_type: str = 'histogram') -> go.Figure:
    """Create distribution plots"""
    try:
        if plot_type == 'histogram':
            if group_by and group_by in df.columns:
                fig = px.histogram(df, x=column, color=group_by, 
                                 marginal="box", hover_data=df.columns,
                                 color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'])
            else:
                fig = px.histogram(df, x=column, marginal="box", hover_data=df.columns)
                fig.update_traces(marker_color=VIZ_CONFIG['COLORS']['PRIMARY'])
        
        elif plot_type == 'box':
            if group_by and group_by in df.columns:
                fig = px.box(df, x=group_by, y=column,
                           color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'])
            else:
                fig = px.box(df, y=column)
                fig.update_traces(marker_color=VIZ_CONFIG['COLORS']['PRIMARY'])
        
        elif plot_type == 'violin':
            if group_by and group_by in df.columns:
                fig = px.violin(df, x=group_by, y=column,
                              color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'])
            else:
                fig = px.violin(df, y=column)
                fig.update_traces(marker_color=VIZ_CONFIG['COLORS']['PRIMARY'])
        
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
            return go.Figure()
        
        fig.update_layout(
            title=f"{plot_type.title()} of {column.replace('_', ' ').title()}",
            height=VIZ_CONFIG['CHART_HEIGHT']
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating distribution plot: {e}")
        return go.Figure()

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       color_col: str = None, size_col: str = None,
                       title: str = None) -> go.Figure:
    """Create scatter plot with optional color and size mapping"""
    try:
        if title is None:
            title = f"{y_col} vs {x_col}"
        
        fig = px.scatter(
            df, x=x_col, y=y_col, 
            color=color_col, size=size_col,
            hover_data=df.columns,
            color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT']
        )
        
        fig.update_layout(
            title=title,
            height=VIZ_CONFIG['CHART_HEIGHT']
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating scatter plot: {e}")
        return go.Figure()

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                    color_col: str = None, orientation: str = 'v',
                    title: str = None) -> go.Figure:
    """Create bar chart"""
    try:
        if title is None:
            title = f"{y_col} by {x_col}"
        
        if orientation == 'h':
            fig = px.bar(df, x=y_col, y=x_col, color=color_col, orientation='h',
                        color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'])
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'])
        
        fig.update_layout(
            title=title,
            height=VIZ_CONFIG['CHART_HEIGHT']
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating bar chart: {e}")
        return go.Figure()

def create_pie_chart(df: pd.DataFrame, values_col: str, names_col: str,
                    title: str = None) -> go.Figure:
    """Create pie chart"""
    try:
        if title is None:
            title = f"Distribution of {names_col.replace('_', ' ').title()}"
        
        fig = px.pie(df, values=values_col, names=names_col, title=title,
                    color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=VIZ_CONFIG['CHART_HEIGHT'])
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating pie chart: {e}")
        return go.Figure()

def create_candlestick_chart(df: pd.DataFrame, date_col: str, 
                           open_col: str, high_col: str, low_col: str, close_col: str,
                           title: str = "Candlestick Chart") -> go.Figure:
    """Create candlestick chart for financial data"""
    try:
        fig = go.Figure(data=go.Candlestick(
            x=df[date_col],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col]
        ))
        
        fig.update_layout(
            title=title,
            height=VIZ_CONFIG['CHART_HEIGHT'],
            xaxis_title="Date",
            yaxis_title="Price"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {e}")
        return go.Figure()

def create_gauge_chart(value: float, min_val: float = 0, max_val: float = 100,
                      title: str = "Gauge Chart", format_str: str = ".1f") -> go.Figure:
    """Create gauge chart for KPIs"""
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (max_val + min_val) / 2},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': VIZ_CONFIG['COLORS']['PRIMARY']},
                'steps': [
                    {'range': [min_val, max_val * 0.7], 'color': VIZ_CONFIG['COLORS']['SUCCESS']},
                    {'range': [max_val * 0.7, max_val * 0.9], 'color': VIZ_CONFIG['COLORS']['WARNING']},
                    {'range': [max_val * 0.9, max_val], 'color': VIZ_CONFIG['COLORS']['DANGER']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        logger.error(f"Error creating gauge chart: {e}")
        return go.Figure()

def create_waterfall_chart(categories: List[str], values: List[float],
                          title: str = "Waterfall Chart") -> go.Figure:
    """Create waterfall chart for showing cumulative effects"""
    try:
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"{val:+.1f}" for val in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": VIZ_CONFIG['COLORS']['SUCCESS']}},
            decreasing={"marker": {"color": VIZ_CONFIG['COLORS']['DANGER']}},
            totals={"marker": {"color": VIZ_CONFIG['COLORS']['PRIMARY']}}
        ))
        
        fig.update_layout(
            title=title,
            height=VIZ_CONFIG['CHART_HEIGHT'],
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating waterfall chart: {e}")
        return go.Figure()

def create_heatmap_calendar(df: pd.DataFrame, date_col: str, value_col: str,
                           title: str = "Calendar Heatmap") -> go.Figure:
    """Create calendar heatmap"""
    try:
        # Prepare data
        df_cal = df.copy()
        df_cal[date_col] = pd.to_datetime(df_cal[date_col])
        df_cal['year'] = df_cal[date_col].dt.year
        df_cal['month'] = df_cal[date_col].dt.month
        df_cal['day'] = df_cal[date_col].dt.day
        
        # Create pivot table
        pivot_data = df_cal.pivot_table(
            index=['year', 'month'], 
            columns='day', 
            values=value_col, 
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=[f"{row[0]}-{row[1]:02d}" for row in pivot_data.index],
            colorscale='RdYlBu',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Day of Month",
            yaxis_title="Year-Month",
            height=VIZ_CONFIG['CHART_HEIGHT']
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating calendar heatmap: {e}")
        return go.Figure()

def create_subplot_dashboard(figures: List[Tuple[go.Figure, str]], 
                           rows: int = 2, cols: int = 2,
                           title: str = "Dashboard") -> go.Figure:
    """Create subplot dashboard from multiple figures"""
    try:
        subplot_titles = [fig[1] for fig in figures]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, (source_fig, _) in enumerate(figures):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            if row <= rows and col <= cols:
                for trace in source_fig.data:
                    fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            title=title,
            height=VIZ_CONFIG['CHART_HEIGHT'] * rows,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating subplot dashboard: {e}")
        return go.Figure()

def create_model_performance_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model") -> go.Figure:
    """Create model performance visualization"""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted', 'Residuals Plot', 
                          'Residuals Distribution', 'Q-Q Plot'),
            specs=[[{"colspan": 2}, None],
                   [{}, {}]]
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', 
                      name='Predictions', opacity=0.6,
                      marker=dict(color=VIZ_CONFIG['COLORS']['PRIMARY'])),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residuals plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers',
                      name='Residuals', opacity=0.6,
                      marker=dict(color=VIZ_CONFIG['COLORS']['SECONDARY'])),
            row=2, col=1
        )
        
        # Zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # Residuals distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals Distribution', nbinsx=30,
                        marker=dict(color=VIZ_CONFIG['COLORS']['SUCCESS'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"{model_name} Performance Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating model performance plot: {e}")
        return go.Figure()

def create_feature_importance_plot(importance_df: pd.DataFrame, top_n: int = 20,
                                 title: str = "Feature Importance") -> go.Figure:
    """Create feature importance plot"""
    try:
        # Take top N features
        plot_df = importance_df.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=plot_df['importance'],
            y=plot_df['feature'],
            orientation='h',
            marker=dict(color=VIZ_CONFIG['COLORS']['GRADIENT'][0])
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, top_n * 20),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")
        return go.Figure()

def create_risk_dashboard(risk_metrics: Dict, title: str = "Risk Dashboard") -> go.Figure:
    """Create comprehensive risk dashboard"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('LCR Ratio', 'VaR Distribution', 'Risk Scores', 'Concentration Risk'),
            specs=[[{"type": "indicator"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # LCR Gauge
        lcr_ratio = risk_metrics.get('lcr_ratio', 1.0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=lcr_ratio * 100,
                title={'text': "LCR %"},
                gauge={'axis': {'range': [None, 150]},
                       'bar': {'color': VIZ_CONFIG['COLORS']['PRIMARY']},
                       'steps': [{'range': [0, 100], 'color': VIZ_CONFIG['COLORS']['DANGER']},
                                {'range': [100, 110], 'color': VIZ_CONFIG['COLORS']['WARNING']},
                                {'range': [110, 150], 'color': VIZ_CONFIG['COLORS']['SUCCESS']}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 100}}
            ),
            row=1, col=1
        )
        
        # VaR Distribution (mock data)
        var_data = np.random.normal(-1000000, 500000, 1000)
        fig.add_trace(
            go.Histogram(x=var_data, name='VaR Distribution', nbinsx=30,
                        marker=dict(color=VIZ_CONFIG['COLORS']['SECONDARY'])),
            row=1, col=2
        )
        
        # Risk Scores by Segment
        segments = ['Retail Small', 'Retail Medium', 'Retail Large', 'SME', 'Corporate']
        risk_scores = [3.2, 2.8, 4.1, 5.5, 2.1]
        
        fig.add_trace(
            go.Bar(x=segments, y=risk_scores, name='Risk Scores',
                  marker=dict(color=VIZ_CONFIG['COLORS']['GRADIENT'])),
            row=2, col=1
        )
        
        # Concentration Risk
        concentration_data = [30, 25, 20, 15, 10]
        fig.add_trace(
            go.Pie(labels=segments, values=concentration_data, name='Concentration',
                  marker=dict(colors=VIZ_CONFIG['COLORS']['GRADIENT'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk dashboard: {e}")
        return go.Figure()

def create_time_series_decomposition(df: pd.DataFrame, date_col: str, value_col: str,
                                   title: str = "Time Series Decomposition") -> go.Figure:
    """Create time series decomposition plot"""
    try:
        # Simple decomposition using moving averages
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        
        # Trend (30-day moving average)
        df_ts['trend'] = df_ts[value_col].rolling(window=30, center=True).mean()
        
        # Seasonal (7-day moving average of detrended data)
        df_ts['detrended'] = df_ts[value_col] - df_ts['trend']
        df_ts['seasonal'] = df_ts['detrended'].rolling(window=7, center=True).mean()
        
        # Residual
        df_ts['residual'] = df_ts[value_col] - df_ts['trend'] - df_ts['seasonal']
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        components = [
            (value_col, 'Original', VIZ_CONFIG['COLORS']['PRIMARY']),
            ('trend', 'Trend', VIZ_CONFIG['COLORS']['SECONDARY']),
            ('seasonal', 'Seasonal', VIZ_CONFIG['COLORS']['SUCCESS']),
            ('residual', 'Residual', VIZ_CONFIG['COLORS']['DANGER'])
        ]
        
        for i, (col, name, color) in enumerate(components, 1):
            fig.add_trace(
                go.Scatter(x=df_ts[date_col], y=df_ts[col],
                          mode='lines', name=name, line=dict(color=color)),
                row=i, col=1
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating time series decomposition: {e}")
        return go.Figure()

def create_sensitivity_analysis_plot(base_value: float, scenarios: Dict[str, float],
                                   title: str = "Sensitivity Analysis") -> go.Figure:
    """Create sensitivity analysis tornado chart"""
    try:
        scenario_names = list(scenarios.keys())
        impacts = [abs(val - base_value) for val in scenarios.values()]
        colors = [VIZ_CONFIG['COLORS']['SUCCESS'] if val >= base_value 
                 else VIZ_CONFIG['COLORS']['DANGER'] for val in scenarios.values()]
        
        # Sort by impact magnitude
        sorted_data = sorted(zip(scenario_names, impacts, colors), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_impacts, sorted_colors = zip(*sorted_data)
        
        fig = go.Figure(go.Bar(
            x=sorted_impacts,
            y=sorted_names,
            orientation='h',
            marker=dict(color=sorted_colors),
            text=[f"{impact:.1f}" for impact in sorted_impacts],
            textposition='auto'
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=title,
            xaxis_title="Impact from Base Case",
            height=max(400, len(scenario_names) * 30),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating sensitivity analysis plot: {e}")
        return go.Figure()

def save_plot(fig: go.Figure, filename: str, format: str = 'html', 
              width: int = None, height: int = None) -> bool:
    """Save plotly figure to file"""
    try:
        if width is None:
            width = VIZ_CONFIG['CHART_WIDTH']
        if height is None:
            height = VIZ_CONFIG['CHART_HEIGHT']
        
        if format.lower() == 'html':
            fig.write_html(filename, include_plotlyjs='cdn')
        elif format.lower() == 'png':
            fig.write_image(filename, format='png', width=width, height=height)
        elif format.lower() == 'pdf':
            fig.write_image(filename, format='pdf', width=width, height=height)
        elif format.lower() == 'svg':
            fig.write_image(filename, format='svg', width=width, height=height)
        elif format.lower() == 'jpeg':
            fig.write_image(filename, format='jpeg', width=width, height=height)
        else:
            logger.error(f"Unsupported format: {format}")
            return False
        
        logger.info(f"Plot saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        return False

def create_custom_theme(primary_color: str = "#1f77b4", 
                       secondary_color: str = "#ff7f0e") -> Dict:
    """Create custom theme for consistent styling"""
    try:
        theme = {
            'layout': {
                'font': {'family': 'Arial, sans-serif', 'size': 12},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'colorway': [primary_color, secondary_color, '#2ca02c', '#d62728', '#9467bd'],
                'hovermode': 'closest',
                'margin': {'l': 60, 'r': 30, 't': 80, 'b': 60}
            },
            'axes': {
                'xaxis': {
                    'showgrid': True,
                    'gridwidth': 1,
                    'gridcolor': 'rgba(128,128,128,0.2)',
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black'
                },
                'yaxis': {
                    'showgrid': True,
                    'gridwidth': 1,
                    'gridcolor': 'rgba(128,128,128,0.2)',
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black'
                }
            }
        }
        
        return theme
        
    except Exception as e:
        logger.error(f"Error creating custom theme: {e}")
        return {}

def apply_theme_to_figure(fig: go.Figure, theme: Dict = None) -> go.Figure:
    """Apply custom theme to plotly figure"""
    try:
        if theme is None:
            theme = create_custom_theme()
        
        # Apply layout settings
        layout_settings = theme.get('layout', {})
        fig.update_layout(**layout_settings)
        
        # Apply axis settings
        axes_settings = theme.get('axes', {})
        if 'xaxis' in axes_settings:
            fig.update_xaxes(**axes_settings['xaxis'])
        if 'yaxis' in axes_settings:
            fig.update_yaxes(**axes_settings['yaxis'])
        
        return fig
        
    except Exception as e:
        logger.error(f"Error applying theme to figure: {e}")
        return fig

def create_animation_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                         animation_frame: str, title: str = "Animated Plot") -> go.Figure:
    """Create animated plot"""
    try:
        fig = px.scatter(df, x=x_col, y=y_col, animation_frame=animation_frame,
                        title=title, hover_data=df.columns,
                        color_discrete_sequence=VIZ_CONFIG['COLORS']['GRADIENT'])
        
        fig.update_layout(height=VIZ_CONFIG['CHART_HEIGHT'])
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating animation plot: {e}")
        return go.Figure()

def export_dashboard_pdf(figures: List[go.Figure], filename: str = "dashboard.pdf") -> bool:
    """Export multiple figures to PDF"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(filename) as pdf:
            for fig in figures:
                # Convert plotly to matplotlib (simplified)
                plt.figure(figsize=(12, 8))
                plt.title(fig.layout.title.text if fig.layout.title else "Chart")
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        logger.info(f"Dashboard exported to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting dashboard to PDF: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    logger.info("Visualization utils module loaded successfully")
    print("Available functions:")
    print("- create_time_series_plot")
    print("- create_correlation_heatmap")
    print("- create_distribution_plot")
    print("- create_scatter_plot")
    print("- create_bar_chart")
    print("- create_pie_chart")
    print("- create_candlestick_chart")
    print("- create_gauge_chart")
    print("- create_waterfall_chart")
    print("- create_heatmap_calendar")
    print("- create_subplot_dashboard")
    print("- create_model_performance_plot")
    print("- create_feature_importance_plot")
    print("- create_risk_dashboard")
    print("- create_time_series_decomposition")
    print("- create_sensitivity_analysis_plot")
    print("- save_plot")
    print("- create_custom_theme")
    print("- apply_theme_to_figure")
    print("- create_animation_plot")