import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from config import MODEL_CONFIG, VALIDATION_CONFIG
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series, 
                             hyperparameter_tuning: bool = True) -> Tuple[RandomForestRegressor, Dict]:
    """Train Random Forest model for deposit flow prediction"""
    try:
        logger.info("Training Random Forest model...")
        
        if hyperparameter_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            rf = RandomForestRegressor(
                random_state=MODEL_CONFIG['RANDOM_FOREST']['random_state'],
                n_jobs=MODEL_CONFIG['RANDOM_FOREST']['n_jobs']
            )
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best RF parameters: {best_params}")
            
        else:
            # Use default parameters from config
            best_model = RandomForestRegressor(**MODEL_CONFIG['RANDOM_FOREST'])
            best_model.fit(X_train, y_train)
            best_params = MODEL_CONFIG['RANDOM_FOREST']
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_info = {
            'model_type': 'RandomForest',
            'best_params': best_params,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'feature_importance': feature_importance,
            'n_features': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        logger.info(f"Random Forest CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std()*2:.4f})")
        return best_model, model_info
        
    except Exception as e:
        logger.error(f"Error training Random Forest model: {e}")
        return None, {}

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                       hyperparameter_tuning: bool = True) -> Tuple[xgb.XGBRegressor, Dict]:
    """Train XGBoost model for deposit flow prediction"""
    try:
        logger.info("Training XGBoost model...")
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(
                random_state=MODEL_CONFIG['XGBOOST']['random_state'],
                n_jobs=MODEL_CONFIG['XGBOOST']['n_jobs']
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best XGBoost parameters: {best_params}")
            
        else:
            # Use default parameters from config
            best_model = xgb.XGBRegressor(**MODEL_CONFIG['XGBOOST'])
            best_model.fit(X_train, y_train)
            best_params = MODEL_CONFIG['XGBOOST']
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_info = {
            'model_type': 'XGBoost',
            'best_params': best_params,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'feature_importance': feature_importance,
            'n_features': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        logger.info(f"XGBoost CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std()*2:.4f})")
        return best_model, model_info
        
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        return None, {}

def evaluate_model_performance(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str = "Model") -> Dict:
    """Evaluate model performance on test set"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        # Directional accuracy (for time series)
        y_test_diff = np.diff(y_test)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff)) * 100
        
        performance_metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'test_samples': len(y_test)
        }
        
        logger.info(f"{model_name} Performance:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model performance: {e}")
        return {}

def create_prediction_intervals(model: Any, X_test: pd.DataFrame, 
                              confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Create prediction intervals using quantile regression approach"""
    try:
        # For ensemble models, use prediction variance
        if hasattr(model, 'estimators_'):
            # Random Forest - use prediction variance from individual trees
            predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Calculate confidence intervals
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
            
        else:
            # For other models, use bootstrap approach
            n_bootstrap = 100
            predictions = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
                X_bootstrap = X_test.iloc[indices]
                pred = model.predict(X_bootstrap)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            alpha = 1 - confidence_level
            lower_bound = np.percentile(predictions, alpha/2 * 100, axis=0)
            upper_bound = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        
        logger.info(f"Created {confidence_level*100}% prediction intervals")
        return lower_bound, upper_bound
        
    except Exception as e:
        logger.error(f"Error creating prediction intervals: {e}")
        return np.array([]), np.array([])

def analyze_feature_importance(model_info: Dict, top_n: int = 20) -> pd.DataFrame:
    """Analyze and visualize feature importance"""
    try:
        if 'feature_importance' not in model_info:
            logger.warning("No feature importance found in model info")
            return pd.DataFrame()
        
        importance_df = model_info['feature_importance'].head(top_n)
        
        # Categorize features
        def categorize_feature(feature_name):
            if any(keyword in feature_name.lower() for keyword in ['rate', 'treasury', 'yield']):
                return 'Interest Rate'
            elif any(keyword in feature_name.lower() for keyword in ['vix', 'volatility']):
                return 'Market Volatility'
            elif any(keyword in feature_name.lower() for keyword in ['segment', 'flow', 'behavioral']):
                return 'Customer Behavior'
            elif any(keyword in feature_name.lower() for keyword in ['month', 'day', 'quarter']):
                return 'Calendar'
            elif any(keyword in feature_name.lower() for keyword in ['lag', 'ma', 'rolling']):
                return 'Technical'
            else:
                return 'Other'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        # Summary by category
        category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        logger.info("Top feature categories by importance:")
        for category, importance in category_importance.items():
            logger.info(f"  {category}: {importance:.4f}")
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")
        return pd.DataFrame()

def validate_model_stability(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                           n_splits: int = 5) -> Dict:
    """Validate model stability across different time periods"""
    try:
        logger.info("Validating model stability...")
        
        # Time series split validation
        split_size = len(X_train) // n_splits
        stability_metrics = []
        
        for i in range(1, n_splits):
            # Use expanding window
            train_end = split_size * (i + 1)
            val_start = train_end
            val_end = min(train_end + split_size, len(X_train))
            
            if val_end > val_start:
                X_val_train = X_train.iloc[:train_end]
                y_val_train = y_train.iloc[:train_end]
                X_val_test = X_train.iloc[val_start:val_end]
                y_val_test = y_train.iloc[val_start:val_end]
                
                # Train model on subset
                temp_model = type(model)(**model.get_params())
                temp_model.fit(X_val_train, y_val_train)
                
                # Evaluate
                val_metrics = evaluate_model_performance(temp_model, X_val_test, y_val_test, f"Split_{i}")
                stability_metrics.append(val_metrics)
        
        if stability_metrics:
            # Calculate stability statistics
            rmse_values = [m['rmse'] for m in stability_metrics]
            r2_values = [m['r2_score'] for m in stability_metrics]
            
            stability_summary = {
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values),
                'rmse_cv': np.std(rmse_values) / np.mean(rmse_values),
                'r2_mean': np.mean(r2_values),
                'r2_std': np.std(r2_values),
                'n_splits': len(stability_metrics)
            }
            
            logger.info(f"Model stability - RMSE CV: {stability_summary['rmse_cv']:.4f}")
            return stability_summary
        else:
            return {}
        
    except Exception as e:
        logger.error(f"Error validating model stability: {e}")
        return {}

def create_segment_specific_models(X_train: pd.DataFrame, y_train: pd.Series,
                                 segment_col: str = 'segment') -> Dict[str, Tuple[Any, Dict]]:
    """Create separate models for each customer segment"""
    try:
        logger.info("Creating segment-specific models...")
        
        segment_models = {}
        
        if segment_col not in X_train.columns:
            logger.warning(f"Segment column {segment_col} not found, creating single model")
            model, info = train_xgboost_model(X_train, y_train, hyperparameter_tuning=False)
            return {'ALL_SEGMENTS': (model, info)}
        
        segments = X_train[segment_col].unique()
        
        for segment in segments:
            logger.info(f"Training model for segment: {segment}")
            
            # Filter data for this segment
            segment_mask = X_train[segment_col] == segment
            X_segment = X_train[segment_mask].drop(columns=[segment_col])
            y_segment = y_train[segment_mask]
            
            if len(X_segment) > 50:  # Minimum samples for training
                # Use XGBoost for segment models (faster for smaller datasets)
                model, info = train_xgboost_model(X_segment, y_segment, hyperparameter_tuning=False)
                segment_models[segment] = (model, info)
                
                logger.info(f"Segment {segment} model trained with {len(X_segment)} samples")
            else:
                logger.warning(f"Insufficient data for segment {segment}: {len(X_segment)} samples")
        
        return segment_models
        
    except Exception as e:
        logger.error(f"Error creating segment-specific models: {e}")
        return {}

def predict_with_segment_models(segment_models: Dict, X_test: pd.DataFrame,
                               segment_col: str = 'segment') -> np.ndarray:
    """Make predictions using segment-specific models"""
    try:
        if segment_col not in X_test.columns:
            # Use single model if available
            if 'ALL_SEGMENTS' in segment_models:
                model, _ = segment_models['ALL_SEGMENTS']
                return model.predict(X_test)
            else:
                logger.error("No segment column and no general model available")
                return np.array([])
        
        predictions = np.full(len(X_test), np.nan)
        
        for segment, (model, _) in segment_models.items():
            segment_mask = X_test[segment_col] == segment
            if segment_mask.any():
                X_segment = X_test[segment_mask].drop(columns=[segment_col])
                segment_predictions = model.predict(X_segment)
                predictions[segment_mask] = segment_predictions
        
        # Handle any missing predictions (segments not seen in training)
        missing_mask = np.isnan(predictions)
        if missing_mask.any():
            logger.warning(f"Missing predictions for {missing_mask.sum()} samples")
            # Use mean prediction as fallback
            predictions[missing_mask] = np.nanmean(predictions)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error predicting with segment models: {e}")
        return np.array([])

def create_behavioral_risk_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict]:
    """Create specialized model for behavioral risk prediction"""
    try:
        logger.info("Creating behavioral risk model...")
        
        # Transform target to risk categories
        y_risk = pd.cut(y_train, bins=[-np.inf, -2*y_train.std(), -0.5*y_train.std(), 
                                       0.5*y_train.std(), 2*y_train.std(), np.inf],
                       labels=['Very High Risk', 'High Risk', 'Normal', 'Low Risk', 'Very Low Risk'])
        
        # Use Random Forest for risk classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score
        
        risk_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        risk_model.fit(X_train, y_risk)
        
        # Cross-validation for classification
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(risk_model, X_train, y_risk, cv=5, scoring='accuracy')
        
        # Feature importance for risk model
        risk_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': risk_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        risk_model_info = {
            'model_type': 'BehavioralRisk',
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'feature_importance': risk_importance,
            'risk_categories': list(y_risk.cat.categories),
            'n_features': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        logger.info(f"Behavioral Risk Model CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        return risk_model, risk_model_info
        
    except Exception as e:
        logger.error(f"Error creating behavioral risk model: {e}")
        return None, {}

def save_trained_models(models_dict: Dict[str, Tuple[Any, Dict]], 
                       model_directory: str = 'data/models/') -> bool:
    """Save trained models and their metadata"""
    try:
        import os
        os.makedirs(model_directory, exist_ok=True)
        
        for model_name, (model, model_info) in models_dict.items():
            # Save model
            model_filename = f"{model_directory}{model_name.lower()}_model.pkl"
            joblib.dump(model, model_filename)
            
            # Save model info
            info_filename = f"{model_directory}{model_name.lower()}_info.pkl"
            joblib.dump(model_info, info_filename)
            
            logger.info(f"Saved {model_name} model to {model_filename}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

def load_trained_models(model_directory: str = 'data/models/') -> Dict[str, Tuple[Any, Dict]]:
    """Load previously trained models"""
    try:
        import os
        import glob
        
        models_dict = {}
        model_files = glob.glob(f"{model_directory}*_model.pkl")
        
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('_model.pkl', '').upper()
            info_file = model_file.replace('_model.pkl', '_info.pkl')
            
            if os.path.exists(info_file):
                model = joblib.load(model_file)
                model_info = joblib.load(info_file)
                models_dict[model_name] = (model, model_info)
                logger.info(f"Loaded {model_name} model from {model_file}")
            else:
                logger.warning(f"Info file not found for {model_name}")
        
        return models_dict
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return {}

def train_all_behavioral_models(X_train: pd.DataFrame, y_train: pd.Series,
                               hyperparameter_tuning: bool = False) -> Dict[str, Tuple[Any, Dict]]:
    """Train all behavioral models and return them in a dictionary"""
    logger.info("=== Training All Behavioral Models ===")
    
    try:
        all_models = {}
        
        # 1. Random Forest Model
        logger.info("1. Training Random Forest...")
        rf_model, rf_info = train_random_forest_model(X_train, y_train, hyperparameter_tuning)
        if rf_model is not None:
            all_models['RANDOM_FOREST'] = (rf_model, rf_info)
        
        # 2. XGBoost Model
        logger.info("2. Training XGBoost...")
        xgb_model, xgb_info = train_xgboost_model(X_train, y_train, hyperparameter_tuning)
        if xgb_model is not None:
            all_models['XGBOOST'] = (xgb_model, xgb_info)
        
        # 3. Segment-specific Models
        logger.info("3. Training Segment Models...")
        segment_models = create_segment_specific_models(X_train, y_train)
        for segment, (model, info) in segment_models.items():
            all_models[f'SEGMENT_{segment}'] = (model, info)
        
        # 4. Behavioral Risk Model
        logger.info("4. Training Behavioral Risk Model...")
        risk_model, risk_info = create_behavioral_risk_model(X_train, y_train)
        if risk_model is not None:
            all_models['BEHAVIORAL_RISK'] = (risk_model, risk_info)
        
        # Save all models
        save_trained_models(all_models)
        
        logger.info(f"=== Trained {len(all_models)} models successfully ===")
        return all_models
        
    except Exception as e:
        logger.error(f"Error training all behavioral models: {e}")
        return {}

def compare_model_performance(models_dict: Dict[str, Tuple[Any, Dict]], 
                            X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compare performance of all trained models"""
    try:
        logger.info("Comparing model performance...")
        
        performance_results = []
        
        for model_name, (model, model_info) in models_dict.items():
            try:
                if 'SEGMENT_' in model_name:
                    # Handle segment models differently
                    continue
                elif model_name == 'BEHAVIORAL_RISK':
                    # Skip risk model for regression comparison
                    continue
                else:
                    # Standard regression models
                    performance = evaluate_model_performance(model, X_test, y_test, model_name)
                    performance['model_type'] = model_info.get('model_type', model_name)
                    performance_results.append(performance)
                    
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        if performance_results:
            performance_df = pd.DataFrame(performance_results)
            performance_df = performance_df.sort_values('rmse')
            
            logger.info("Model Performance Comparison:")
            logger.info(performance_df[['model_name', 'rmse', 'r2_score', 'mape']].to_string())
            
            return performance_df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error comparing model performance: {e}")
        return pd.DataFrame()

# Main execution functions
def main_model_training_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                hyperparameter_tuning: bool = False) -> Dict[str, Any]:
    """Main pipeline for training and evaluating behavioral models"""
    logger.info("=== Starting Behavioral Model Training Pipeline ===")
    
    try:
        # Train all models
        all_models = train_all_behavioral_models(X_train, y_train, hyperparameter_tuning)
        
        if not all_models:
            logger.error("No models were trained successfully")
            return {}
        
        # Compare performance
        performance_comparison = compare_model_performance(all_models, X_test, y_test)
        
        # Get best model
        best_model_name = performance_comparison.iloc[0]['model_name'] if not performance_comparison.empty else 'XGBOOST'
        best_model, best_model_info = all_models.get(best_model_name, (None, {}))
        
        # Analyze feature importance for best model
        feature_analysis = analyze_feature_importance(best_model_info)
        
        # Validate model stability
        stability_analysis = validate_model_stability(best_model, X_train, y_train) if best_model else {}
        
        # Create prediction intervals for best model
        if best_model:
            lower_bounds, upper_bounds = create_prediction_intervals(best_model, X_test)
        else:
            lower_bounds, upper_bounds = np.array([]), np.array([])
        
        pipeline_results = {
            'trained_models': all_models,
            'performance_comparison': performance_comparison,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_model_info': best_model_info,
            'feature_analysis': feature_analysis,
            'stability_analysis': stability_analysis,
            'prediction_intervals': {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds
            }
        }
        
        logger.info("=== Behavioral Model Training Pipeline Complete ===")
        logger.info(f"Best model: {best_model_name}")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("Behavioral models module loaded successfully")
    print("Available functions:")
    print("- train_random_forest_model")
    print("- train_xgboost_model") 
    print("- evaluate_model_performance")
    print("- create_prediction_intervals")
    print("- analyze_feature_importance")
    print("- validate_model_stability")
    print("- create_segment_specific_models")
    print("- create_behavioral_risk_model")
    print("- train_all_behavioral_models")
    print("- main_model_training_pipeline")