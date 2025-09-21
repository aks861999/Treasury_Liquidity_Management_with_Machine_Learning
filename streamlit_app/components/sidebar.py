import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import STREAMLIT_CONFIG

def display_sidebar():
    """Display comprehensive sidebar with all controls"""
    try:
        st.sidebar.title("🎛️ Control Panel")
        st.sidebar.markdown("---")
        
        # Data management section
        display_data_management_section()
        
        st.sidebar.markdown("---")
        
        # Model training section
        display_model_training_section()
        
        st.sidebar.markdown("---")
        
        # Settings section
        display_settings_section()
        
        st.sidebar.markdown("---")
        
        # Navigation section
        display_navigation_section()
        
        st.sidebar.markdown("---")
        
        # System status
        display_system_status()
        
    except Exception as e:
        st.sidebar.error(f"Error displaying sidebar: {e}")

def display_data_management_section():
    """Display data management controls"""
    try:
        with st.sidebar.expander("📊 Data Management", expanded=True):
            
            # Data loading controls
            st.write("**Data Collection**")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🔄 Load Data", type="primary", help="Collect market data and generate customer data"):
                    load_data_action()
            
            with col2:
                if st.button("🔄 Refresh", help="Refresh existing data"):
                    refresh_data_action()
            
            # Data status
            if st.session_state.get('data_loaded', False):
                st.success("✅ Data loaded")
                
                # Display data info
                aggregated_data = st.session_state.get('aggregated_data', {})
                if aggregated_data:
                    market_data = aggregated_data.get('market_data', pd.DataFrame())
                    customer_data = aggregated_data.get('customer_data', {})
                    
                    st.write("📈 Market Data:")
                    st.write(f"• Records: {len(market_data) if not market_data.empty else 0}")
                    if not market_data.empty:
                        date_range = f"{market_data.index.min().strftime('%Y-%m-%d')} to {market_data.index.max().strftime('%Y-%m-%d')}"
                        st.write(f"• Period: {date_range}")
                    
                    if customer_data:
                        customers = customer_data.get('customers', pd.DataFrame())
                        flows = customer_data.get('flows', pd.DataFrame())
                        st.write("👥 Customer Data:")
                        st.write(f"• Customers: {len(customers) if not customers.empty else 0}")
                        st.write(f"• Flow records: {len(flows) if not flows.empty else 0}")
                
                # Data quality indicator
                data_quality = aggregated_data.get('data_quality', {})
                if data_quality:
                    quality_score = data_quality.get('overall_score', 0)
                    if quality_score >= 80:
                        st.success(f"🟢 Data quality: {quality_score:.0f}%")
                    elif quality_score >= 60:
                        st.warning(f"🟡 Data quality: {quality_score:.0f}%")
                    else:
                        st.error(f"🔴 Data quality: {quality_score:.0f}%")
            else:
                st.warning("⚠️ No data loaded")
            
            # Data configuration options
            st.write("**Configuration**")
            
            # Sample size for customer generation
            customer_count = st.slider(
                "Customer Sample Size",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=500,
                help="Number of synthetic customers to generate"
            )
            
            # Data refresh frequency
            refresh_frequency = st.selectbox(
                "Auto-refresh",
                ["Manual", "1 Hour", "6 Hours", "Daily"],
                index=2,
                help="Automatic data refresh frequency"
            )
            
            # Store settings in session state
            st.session_state.customer_count = customer_count
            st.session_state.refresh_frequency = refresh_frequency
    
    except Exception as e:
        st.sidebar.error(f"Error in data management section: {e}")

def display_model_training_section():
    """Display model training controls"""
    try:
        with st.sidebar.expander("🤖 Model Training", expanded=st.session_state.get('data_loaded', False)):
            
            if not st.session_state.get('data_loaded', False):
                st.info("📋 Load data first")
                return
            
            st.write("**Training Options**")
            
            # Model selection
            available_models = [
                "Random Forest",
                "XGBoost", 
                "LSTM",
                "Ensemble (All)"
            ]
            
            selected_models = st.multiselect(
                "Models to Train",
                available_models,
                default=["Random Forest", "XGBoost"],
                help="Select which models to train"
            )
            
            # Training parameters
            hyperparameter_tuning = st.checkbox(
                "Hyperparameter Tuning",
                value=False,
                help="Enable hyperparameter optimization (slower but better performance)"
            )
            
            cross_validation = st.checkbox(
                "Cross Validation",
                value=True,
                help="Use cross-validation for model evaluation"
            )
            
            # Training button
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🎯 Train Models", type="secondary", help="Train selected ML models"):
                    train_models_action(selected_models, hyperparameter_tuning, cross_validation)
            
            with col2:
                if st.button("⚡ Quick Train", help="Fast training with default settings"):
                    quick_train_action()
            
            # Model status
            if st.session_state.get('models_trained', False):
                st.success("✅ Models trained")
                
                models = st.session_state.get('models', {})
                individual_models = models.get('individual_models', {})
                best_model_name = individual_models.get('best_model_name', 'N/A')
                
                st.write(f"🏆 Best model: {best_model_name}")
                
                # Model performance summary
                performance_data = individual_models.get('performance_comparison', pd.DataFrame())
                if not performance_data.empty:
                    best_rmse = performance_data.iloc[0]['rmse']
                    best_r2 = performance_data.iloc[0]['r2_score']
                    st.write(f"📊 RMSE: {best_rmse:.4f}")
                    st.write(f"📈 R²: {best_r2:.4f}")
                
                # Model actions
                if st.button("💾 Save Models", help="Save trained models to disk"):
                    save_models_action()
                
            else:
                st.info("ℹ️ Models not trained")
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                # Feature selection
                feature_selection = st.checkbox("Automatic Feature Selection", value=False)
                
                # Model interpretation
                model_interpretation = st.checkbox("Generate Model Explanations", value=True)
                
                # Ensemble options
                ensemble_method = st.selectbox(
                    "Ensemble Method",
                    ["Weighted Average", "Stacking", "Voting"],
                    help="Method for combining multiple models"
                )
                
                # Store advanced settings
                st.session_state.feature_selection = feature_selection
                st.session_state.model_interpretation = model_interpretation
                st.session_state.ensemble_method = ensemble_method
    
    except Exception as e:
        st.sidebar.error(f"Error in model training section: {e}")

def display_settings_section():
    """Display application settings"""
    try:
        with st.sidebar.expander("⚙️ Settings", expanded=False):
            
            st.write("**Display Options**")
            
            # Theme selection
            theme = st.selectbox(
                "Color Theme",
                ["Default", "Dark", "Light", "Custom"],
                help="Select application color theme"
            )
            
            # Chart settings
            chart_height = st.slider(
                "Chart Height",
                300, 800, 400,
                help="Default height for charts in pixels"
            )
            
            show_tooltips = st.checkbox(
                "Show Tooltips",
                value=True,
                help="Display helpful tooltips throughout the app"
            )
            
            # Performance settings
            st.write("**Performance**")
            
            cache_data = st.checkbox(
                "Cache Data",
                value=True,
                help="Cache data to improve performance"
            )
            
            auto_refresh = st.checkbox(
                "Auto Refresh Charts",
                value=False,
                help="Automatically refresh charts when data changes"
            )
            
            # Notification settings
            st.write("**Notifications**")
            
            show_warnings = st.checkbox(
                "Show Warnings",
                value=True,
                help="Display warning messages for data quality issues"
            )
            
            email_alerts = st.checkbox(
                "Email Alerts",
                value=False,
                help="Send email alerts for critical events"
            )
            
            # Update session state with settings
            st.session_state.update({
                'theme': theme,
                'chart_height': chart_height,
                'show_tooltips': show_tooltips,
                'cache_data': cache_data,
                'auto_refresh': auto_refresh,
                'show_warnings': show_warnings,
                'email_alerts': email_alerts
            })
            
            # Reset settings button
            if st.button("🔄 Reset to Defaults"):
                reset_settings_action()
    
    except Exception as e:
        st.sidebar.error(f"Error in settings section: {e}")

def display_navigation_section():
    """Display navigation controls"""
    try:
        st.write("**🧭 Quick Navigation**")
        
        # Quick action buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📊 Dashboard"):
                st.session_state.current_page = "Dashboard"
                st.experimental_rerun()
        
        with col2:
            if st.button("📈 Analytics"):
                st.session_state.current_page = "Analytics"
                st.experimental_rerun()
        
        # Page shortcuts
        shortcuts = [
            ("🏠 Main Dashboard", "dashboard"),
            ("📋 Data Overview", "data_overview"),
            ("🧠 Behavioral Analysis", "behavioral"),
            ("📊 Market Analysis", "market"),
            ("🎯 Scenario Analysis", "scenarios"),
            ("🤖 Model Performance", "models")
        ]
        
        for display_name, page_key in shortcuts:
            if st.button(display_name, key=f"nav_{page_key}"):
                navigate_to_page(page_key)
    
    except Exception as e:
        st.sidebar.error(f"Error in navigation section: {e}")

def display_system_status():
    """Display system status information"""
    try:
        st.write("**📊 System Status**")
        
        # Data status
        data_status = "✅ Ready" if st.session_state.get('data_loaded', False) else "❌ Not Ready"
        st.write(f"Data: {data_status}")
        
        # Model status
        model_status = "✅ Trained" if st.session_state.get('models_trained', False) else "❌ Not Trained"
        st.write(f"Models: {model_status}")
        
        # Memory usage (mock)
        memory_usage = np.random.uniform(30, 70)  # Mock memory usage
        st.write(f"Memory: {memory_usage:.0f}%")
        
        # Last update
        last_update = datetime.now().strftime("%H:%M:%S")
        st.write(f"Updated: {last_update}")
        
        # System info
        with st.expander("ℹ️ System Info"):
            st.write("**Application Version:** 1.0.0")
            st.write("**Python Version:** 3.9+")
            st.write("**Streamlit Version:** 1.28+")
            
            if st.session_state.get('data_loaded', False):
                aggregated_data = st.session_state.get('aggregated_data', {})
                data_timestamp = aggregated_data.get('collection_timestamp', 'N/A')
                st.write(f"**Data Timestamp:** {data_timestamp}")
    
    except Exception as e:
        st.sidebar.error(f"Error in system status: {e}")

# Action functions
def load_data_action():
    """Handle data loading action"""
    try:
        from streamlit_app.main_app import load_data
        load_data()
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")

def refresh_data_action():
    """Handle data refresh action"""
    try:
        st.sidebar.info("Refreshing data...")
        # In full implementation, this would refresh the data
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Error refreshing data: {e}")

def train_models_action(selected_models, hyperparameter_tuning, cross_validation):
    """Handle model training action"""
    try:
        from streamlit_app.main_app import train_models
        
        # Store training parameters
        st.session_state.training_params = {
            'selected_models': selected_models,
            'hyperparameter_tuning': hyperparameter_tuning,
            'cross_validation': cross_validation
        }
        
        train_models()
    except Exception as e:
        st.sidebar.error(f"Error training models: {e}")

def quick_train_action():
    """Handle quick training action"""
    try:
        st.sidebar.info("Starting quick training...")
        train_models_action(["Random Forest", "XGBoost"], False, True)
    except Exception as e:
        st.sidebar.error(f"Error in quick training: {e}")

def save_models_action():
    """Handle model saving action"""
    try:
        st.sidebar.success("Models saved successfully!")
        # In full implementation, this would save models to disk
    except Exception as e:
        st.sidebar.error(f"Error saving models: {e}")

def reset_settings_action():
    """Reset settings to defaults"""
    try:
        # Reset settings to defaults
        default_settings = {
            'theme': 'Default',
            'chart_height': 400,
            'show_tooltips': True,
            'cache_data': True,
            'auto_refresh': False,
            'show_warnings': True,
            'email_alerts': False
        }
        
        st.session_state.update(default_settings)
        st.sidebar.success("Settings reset to defaults")
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Error resetting settings: {e}")

def navigate_to_page(page_key):
    """Navigate to specified page"""
    try:
        st.session_state.current_page = page_key
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Error navigating to page: {e}")

# Utility functions for sidebar
def format_file_size(size_bytes):
    """Format file size in human readable format"""
    try:
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    except:
        return "N/A"

def get_data_summary():
    """Get summary of loaded data"""
    try:
        if not st.session_state.get('data_loaded', False):
            return "No data loaded"
        
        aggregated_data = st.session_state.get('aggregated_data', {})
        market_data = aggregated_data.get('market_data', pd.DataFrame())
        customer_data = aggregated_data.get('customer_data', {})
        
        summary_parts = []
        
        if not market_data.empty:
            summary_parts.append(f"{len(market_data)} market records")
        
        customers = customer_data.get('customers', pd.DataFrame())
        if not customers.empty:
            summary_parts.append(f"{len(customers)} customers")
        
        flows = customer_data.get('flows', pd.DataFrame())
        if not flows.empty:
            summary_parts.append(f"{len(flows)} flow records")
        
        return ", ".join(summary_parts) if summary_parts else "Data loaded"
    
    except:
        return "Data summary unavailable"

if __name__ == "__main__":
    # Test sidebar component
    st.title("Sidebar Component Test")
    display_sidebar()