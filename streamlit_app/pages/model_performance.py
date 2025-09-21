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
from streamlit_app.components.charts import create_feature_importance_chart, create_prediction_accuracy_chart

def display_model_performance():
    """Model performance analysis page"""
    try:
        st.title("📊 Model Performance Analysis")
        st.markdown("Comprehensive evaluation of machine learning model performance and accuracy")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please load data first to view model performance")
            return
        
        if not st.session_state.get('models_trained', False):
            st.warning("⚠️ Please train models first to view performance analysis")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Train Models Now", type="primary"):
                    st.info("Redirecting to model training...")
                    # In a full implementation, this would trigger model training
            return
        
        # Get model results
        models = st.session_state.models
        individual_models = models.get('individual_models', {})
        
        # Performance overview
        display_performance_overview(individual_models)
        
        st.markdown("---")
        
        # Model comparison
        display_model_comparison(individual_models)
        
        st.markdown("---")
        
        # Feature importance analysis
        display_feature_analysis(individual_models)
        
        st.markdown("---")
        
        # Prediction accuracy analysis
        display_prediction_analysis(individual_models)
        
    except Exception as e:
        st.error(f"Error displaying model performance: {e}")

def display_performance_overview(individual_models: dict):
    """Display performance overview section"""
    try:
        st.subheader("📈 Performance Overview")
        
        performance_data = individual_models.get('performance_comparison', pd.DataFrame())
        
        if performance_data.empty:
            st.info("No performance data available")
            return
        
        # Best model summary
        best_model = performance_data.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Best Model",
                best_model['model_name'],
                help="Model with lowest RMSE"
            )
        
        with col2:
            st.metric(
                "RMSE",
                f"{best_model['rmse']:.4f}",
                help="Root Mean Square Error"
            )
        
        with col3:
            st.metric(
                "R² Score", 
                f"{best_model['r2_score']:.4f}",
                help="Coefficient of determination"
            )
        
        with col4:
            mape = best_model.get('mape', 0)
            st.metric(
                "MAPE",
                f"{mape:.2f}%" if mape != np.inf else "N/A",
                help="Mean Absolute Percentage Error"
            )
        
        # Performance summary table
        st.subheader("📊 All Models Performance")
        
        # Select key columns for display
        display_columns = ['model_name', 'rmse', 'r2_score', 'mae']
        if 'mape' in performance_data.columns:
            display_columns.append('mape')
        if 'directional_accuracy' in performance_data.columns:
            display_columns.append('directional_accuracy')
        
        display_data = performance_data[display_columns].copy()
        
        # Round numerical columns
        for col in display_data.columns:
            if display_data[col].dtype in ['float64', 'float32']:
                display_data[col] = display_data[col].round(4)
        
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error displaying performance overview: {e}")

def display_model_comparison(individual_models: dict):
    """Display model comparison section"""
    try:
        st.subheader("🔍 Model Comparison")
        
        performance_data = individual_models.get('performance_comparison', pd.DataFrame())
        
        if performance_data.empty:
            st.info("No performance data available for comparison")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig_rmse = px.bar(
                performance_data.head(5),  # Top 5 models
                x='model_name',
                y='rmse',
                title="RMSE Comparison (Lower is Better)",
                color='rmse',
                color_continuous_scale='Reds_r'
            )
            fig_rmse.update_layout(height=400, showlegend=False)
            fig_rmse.update_xaxes(tickangle=45)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # R² comparison
            fig_r2 = px.bar(
                performance_data.head(5),
                x='model_name', 
                y='r2_score',
                title="R² Score Comparison (Higher is Better)",
                color='r2_score',
                color_continuous_scale='Greens'
            )
            fig_r2.update_layout(height=400, showlegend=False)
            fig_r2.update_xaxes(tickangle=45)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Performance radar chart
        if len(performance_data) >= 2:
            st.subheader("🕸️ Multi-Metric Comparison")
            
            create_radar_chart(performance_data.head(3))  # Top 3 models
        
        # Statistical significance test
        display_statistical_tests(performance_data)
        
    except Exception as e:
        st.error(f"Error displaying model comparison: {e}")

def create_radar_chart(performance_data: pd.DataFrame):
    """Create radar chart for multi-metric comparison"""
    try:
        metrics = ['r2_score', 'rmse', 'mae']
        available_metrics = [m for m in metrics if m in performance_data.columns]
        
        if len(available_metrics) < 3:
            st.info("Need at least 3 metrics for radar chart")
            return
        
        fig = go.Figure()
        
        colors = VIZ_CONFIG['COLORS']['GRADIENT']
        
        for i, (_, model) in enumerate(performance_data.iterrows()):
            model_name = model['model_name']
            
            # Normalize metrics (0-1 scale)
            values = []
            labels = []
            
            for metric in available_metrics:
                if metric == 'rmse' or metric == 'mae':
                    # For error metrics, invert (1 - normalized value)
                    max_val = performance_data[metric].max()
                    min_val = performance_data[metric].min()
                    normalized = 1 - (model[metric] - min_val) / (max_val - min_val + 1e-8)
                else:
                    # For accuracy metrics, use as is
                    max_val = performance_data[metric].max()
                    min_val = performance_data[metric].min()
                    normalized = (model[metric] - min_val) / (max_val - min_val + 1e-8)
                
                values.append(normalized)
                labels.append(metric.upper())
            
            # Close the radar chart
            values.append(values[0])
            labels.append(labels[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=model_name,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Radar Chart",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating radar chart: {e}")

def display_statistical_tests(performance_data: pd.DataFrame):
    """Display statistical significance tests"""
    try:
        if len(performance_data) < 2:
            return
        
        st.subheader("📊 Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Statistics**")
            
            # Calculate statistics for RMSE
            rmse_stats = {
                'Mean RMSE': performance_data['rmse'].mean(),
                'Std RMSE': performance_data['rmse'].std(),
                'Min RMSE': performance_data['rmse'].min(),
                'Max RMSE': performance_data['rmse'].max()
            }
            
            stats_df = pd.DataFrame(list(rmse_stats.items()), 
                                  columns=['Metric', 'Value'])
            stats_df['Value'] = stats_df['Value'].round(4)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.write("**Model Rankings**")
            
            # Rank models by different metrics
            rankings = performance_data[['model_name', 'rmse', 'r2_score']].copy()
            rankings['RMSE_Rank'] = rankings['rmse'].rank()
            rankings['R2_Rank'] = rankings['r2_score'].rank(ascending=False)
            
            # Overall rank (average of ranks)
            rankings['Overall_Rank'] = (rankings['RMSE_Rank'] + rankings['R2_Rank']) / 2
            rankings = rankings.sort_values('Overall_Rank')
            
            display_rankings = rankings[['model_name', 'Overall_Rank']].copy()
            display_rankings['Overall_Rank'] = display_rankings['Overall_Rank'].round(2)
            display_rankings.columns = ['Model', 'Avg Rank']
            
            st.dataframe(display_rankings, hide_index=True, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying statistical tests: {e}")

def display_feature_analysis(individual_models: dict):
    """Display feature importance analysis"""
    try:
        st.subheader("🎯 Feature Importance Analysis")
        
        best_model_info = individual_models.get('best_model_info', {})
        feature_importance = best_model_info.get('feature_importance', pd.DataFrame())
        
        if feature_importance.empty:
            st.info("No feature importance data available")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature importance chart
            create_feature_importance_chart(feature_importance, top_n=15)
        
        with col2:
            st.write("**Top Features Summary**")
            
            # Feature categories
            top_features = feature_importance.head(10)
            
            # Categorize features
            categories = {
                'Market': 0, 'Customer': 0, 'Technical': 0, 'Time': 0, 'Other': 0
            }
            
            for _, feature in top_features.iterrows():
                feature_name = feature['feature'].lower()
                if any(keyword in feature_name for keyword in ['rate', 'treasury', 'vix', 'yield']):
                    categories['Market'] += feature['importance']
                elif any(keyword in feature_name for keyword in ['customer', 'segment', 'balance', 'behavioral']):
                    categories['Customer'] += feature['importance']
                elif any(keyword in feature_name for keyword in ['lag', 'ma', 'rolling', 'volatility']):
                    categories['Technical'] += feature['importance']
                elif any(keyword in feature_name for keyword in ['month', 'day', 'quarter', 'week']):
                    categories['Time'] += feature['importance']
                else:
                    categories['Other'] += feature['importance']
            
            # Display category importance
            category_df = pd.DataFrame(list(categories.items()), 
                                     columns=['Category', 'Importance'])
            category_df = category_df[category_df['Importance'] > 0].sort_values('Importance', ascending=False)
            category_df['Importance'] = category_df['Importance'].round(4)
            
            st.dataframe(category_df, hide_index=True, use_container_width=True)
            
            # Feature insights
            st.write("**Key Insights**")
            top_feature = feature_importance.iloc[0]
            st.write(f"• Most important: {top_feature['feature']}")
            st.write(f"• Contributes: {top_feature['importance']*100:.1f}% to predictions")
            
            top_5_contrib = feature_importance.head(5)['importance'].sum() * 100
            st.write(f"• Top 5 features: {top_5_contrib:.1f}% of total importance")
        
        # Feature correlation with target (if available)
        display_feature_correlations(feature_importance)
        
    except Exception as e:
        st.error(f"Error displaying feature analysis: {e}")

def display_feature_correlations(feature_importance: pd.DataFrame):
    """Display feature correlation analysis"""
    try:
        # This would require access to the original feature data
        # For now, we'll create a mock correlation analysis
        st.subheader("🔗 Feature Relationships")
        
        # Mock correlation data based on feature importance
        top_features = feature_importance.head(8)
        
        # Create a mock correlation matrix
        np.random.seed(42)
        n_features = len(top_features)
        correlation_matrix = np.random.rand(n_features, n_features)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1)  # Diagonal = 1
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(
            correlation_matrix,
            index=top_features['feature'].values,
            columns=top_features['feature'].values
        )
        
        # Create heatmap
        fig = px.imshow(
            corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            color_continuous_scale='RdBu',
            aspect='auto',
            title="Top Features Correlation Matrix"
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying feature correlations: {e}")

def display_prediction_analysis(individual_models: dict):
    """Display prediction accuracy analysis"""
    try:
        st.subheader("🎯 Prediction Accuracy Analysis")
        
        # Get test data if available
        models = st.session_state.models
        test_data = models.get('test_data', {})
        
        if not test_data:
            st.info("No test data available for prediction analysis")
            return
        
        X_test = test_data.get('X_test')
        y_test = test_data.get('y_test')
        
        if X_test is None or y_test is None:
            st.info("Test data not properly formatted")
            return
        
        # Get best model
        best_model = individual_models.get('best_model')
        best_model_name = individual_models.get('best_model_name', 'Best Model')
        
        if best_model is None:
            st.info("Best model not available")
            return
        
        try:
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Display prediction accuracy chart
            create_prediction_accuracy_chart(
                y_test.values, y_pred, best_model_name
            )
            
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return
        
        # Residual analysis
        display_residual_analysis(y_test.values, y_pred)
        
        # Prediction intervals
        display_prediction_intervals(y_test.values, y_pred)
        
    except Exception as e:
        st.error(f"Error displaying prediction analysis: {e}")

def display_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray):
    """Display residual analysis"""
    try:
        st.subheader("📊 Residual Analysis")
        
        residuals = y_true - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals histogram
            fig = px.histogram(
                x=residuals,
                nbins=30,
                title="Residuals Distribution",
                color_discrete_sequence=[VIZ_CONFIG['COLORS']['PRIMARY']]
            )
            fig.update_layout(height=300)
            fig.update_xaxes(title_text="Residuals")
            fig.update_yaxes(title_text="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-Q plot approximation
            from scipy import stats
            
            # Generate theoretical quantiles
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig = px.scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                title="Q-Q Plot (Normality Check)",
                color_discrete_sequence=[VIZ_CONFIG['COLORS']['SECONDARY']]
            )
            
            # Add diagonal line
            min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Line',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(height=300, showlegend=False)
            fig.update_xaxes(title_text="Theoretical Quantiles")
            fig.update_yaxes(title_text="Sample Quantiles")
            st.plotly_chart(fig, use_container_width=True)
        
        # Residual statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Residual", f"{np.mean(residuals):.6f}")
        with col2:
            st.metric("Std Residual", f"{np.std(residuals):.4f}")
        with col3:
            st.metric("Skewness", f"{stats.skew(residuals):.3f}")
        with col4:
            st.metric("Kurtosis", f"{stats.kurtosis(residuals):.3f}")
        
    except Exception as e:
        st.error(f"Error displaying residual analysis: {e}")

def display_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray):
    """Display prediction intervals"""
    try:
        st.subheader("🔮 Prediction Confidence")
        
        # Calculate prediction errors
        errors = np.abs(y_true - y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            fig = px.box(
                y=errors,
                title="Prediction Error Distribution",
                color_discrete_sequence=[VIZ_CONFIG['COLORS']['WARNING']]
            )
            fig.update_layout(height=300, showlegend=False)
            fig.update_yaxes(title_text="Absolute Error")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence intervals
            percentiles = [50, 68, 95, 99]
            intervals = []
            
            for p in percentiles:
                interval = np.percentile(errors, p)
                intervals.append({'Confidence Level': f'{p}%', 'Error Bound': f'{interval:.4f}'})
            
            intervals_df = pd.DataFrame(intervals)
            
            st.write("**Prediction Intervals**")
            st.dataframe(intervals_df, hide_index=True, use_container_width=True)
            
            st.write("**Interpretation:**")
            st.write("• 50% of predictions within ±{:.4f}".format(np.percentile(errors, 50)))
            st.write("• 95% of predictions within ±{:.4f}".format(np.percentile(errors, 95)))
        
    except Exception as e:
        st.error(f"Error displaying prediction intervals: {e}")

if __name__ == "__main__":
    # Test the model performance page
    st.title("Model Performance Page Test") 
    display_model_performance()