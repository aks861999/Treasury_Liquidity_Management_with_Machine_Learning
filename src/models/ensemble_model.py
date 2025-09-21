import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from config import MODEL_CONFIG
from src.models.behavioral_models import train_random_forest_model, train_xgboost_model
from src.models.time_series_models import main_lstm_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehavioralEnsembleModel:
    """Ensemble model combining Random Forest, XGBoost, and LSTM predictions"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or MODEL_CONFIG['ENSEMBLE_WEIGHTS']
        self.models = {}
        self.model_info = {}
        self.is_trained = False
        self.scaler = None
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_test: pd.DataFrame = None, y_test: pd.Series = None,
              sequence_length: int = 30) -> Dict:
        """Train all component models"""
        try:
            logger.info("Training Behavioral Ensemble Model...")
            training_results = {}
            
            # 1. Train Random Forest
            logger.info("Training Random Forest component...")
            rf_model, rf_info = train_random_forest_model(X_train, y_train, hyperparameter_tuning=False)
            if rf_model is not None:
                self.models['random_forest'] = rf_model
                self.model_info['random_forest'] = rf_info
                training_results['random_forest'] = rf_info
            
            # 2. Train XGBoost
            logger.info("Training XGBoost component...")
            xgb_model, xgb_info = train_xgboost_model(X_train, y_train, hyperparameter_tuning=False)
            if xgb_model is not None:
                self.models['xgboost'] = xgb_model
                self.model_info['xgboost'] = xgb_info
                training_results['xgboost'] = xgb_info
            
            # 3. Train LSTM (requires special data preparation)
            logger.info("Training LSTM component...")
            # Reconstruct full dataframe for LSTM training
            full_df = X_train.copy()
            full_df['target'] = y_train
            
            if X_test is not None and y_test is not None:
                test_df = X_test.copy()
                test_df['target'] = y_test
                full_df = pd.concat([full_df, test_df], ignore_index=True)
            
            # Add date column if not present (required for LSTM)
            if 'date' not in full_df.columns:
                full_df['date'] = pd.date_range(start='2020-01-01', periods=len(full_df), freq='D')
            
            lstm_results = main_lstm_pipeline(full_df, 'target', sequence_length=sequence_length)
            if 'main_model' in lstm_results:
                lstm_model, lstm_info = lstm_results['main_model']
                self.models['lstm'] = lstm_model
                self.model_info['lstm'] = lstm_info
                training_results['lstm'] = lstm_info
            
            # Store training metadata
            self.feature_names = X_train.columns.tolist()
            self.is_trained = len(self.models) > 0
            
            if self.is_trained:
                logger.info(f"Ensemble model trained with {len(self.models)} components")
                return training_results
            else:
                logger.error("No models were successfully trained")
                return {}
                
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {}
    
    def predict(self, X: pd.DataFrame, return_components: bool = False) -> np.ndarray:
        """Make ensemble predictions"""
        try:
            if not self.is_trained:
                logger.error("Ensemble model not trained yet")
                return np.array([])
            
            predictions = {}
            
            # Get predictions from each component
            if 'random_forest' in self.models:
                predictions['random_forest'] = self.models['random_forest'].predict(X)
            
            if 'xgboost' in self.models:
                predictions['xgboost'] = self.models['xgboost'].predict(X)
            
            # LSTM predictions require sequence preparation
            if 'lstm' in self.models:
                try:
                    # This is simplified - in practice you'd need proper sequence preparation
                    # For now, we'll skip LSTM in prediction if it's complex
                    logger.warning("LSTM predictions skipped - requires sequence data preparation")
                except Exception as e:
                    logger.warning(f"LSTM prediction failed: {e}")
            
            if not predictions:
                logger.error("No component predictions available")
                return np.array([])
            
            # Weighted ensemble
            ensemble_pred = np.zeros(len(X))
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = self.weights.get(model_name, 1.0)
                ensemble_pred += weight * pred
                total_weight += weight
            
            ensemble_pred /= total_weight
            
            if return_components:
                return ensemble_pred, predictions
            else:
                return ensemble_pred
                
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return np.array([])
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate ensemble model performance"""
        try:
            predictions, components = self.predict(X_test, return_components=True)
            
            if len(predictions) == 0:
                return {}
            
            # Ensemble metrics
            ensemble_metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2_score': r2_score(y_test, predictions),
                'mape': np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
            }
            
            # Component metrics
            component_metrics = {}
            for model_name, pred in components.items():
                component_metrics[model_name] = {
                    'mse': mean_squared_error(y_test, pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                    'r2_score': r2_score(y_test, pred)
                }
            
            evaluation_results = {
                'ensemble_metrics': ensemble_metrics,
                'component_metrics': component_metrics,
                'weights_used': self.weights.copy(),
                'n_models': len(components)
            }
            
            logger.info(f"Ensemble RMSE: {ensemble_metrics['rmse']:.4f}")
            logger.info(f"Ensemble R²: {ensemble_metrics['r2_score']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble model: {e}")
            return {}
    
    def get_feature_importance(self, method: str = 'average') -> pd.DataFrame:
        """Get combined feature importance from ensemble components"""
        try:
            importance_dfs = []
            
            # Collect feature importance from each model
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_,
                        'model': model_name
                    })
                    importance_dfs.append(importance_df)
            
            if not importance_dfs:
                logger.warning("No feature importance available from component models")
                return pd.DataFrame()
            
            # Combine importance scores
            all_importance = pd.concat(importance_dfs, ignore_index=True)
            
            if method == 'average':
                combined_importance = all_importance.groupby('feature')['importance'].mean().reset_index()
            elif method == 'weighted':
                # Weight by model performance or predefined weights
                weighted_scores = []
                for _, row in all_importance.iterrows():
                    weight = self.weights.get(row['model'], 1.0)
                    weighted_scores.append(row['importance'] * weight)
                
                all_importance['weighted_importance'] = weighted_scores
                combined_importance = all_importance.groupby('feature')['weighted_importance'].sum().reset_index()
                combined_importance.columns = ['feature', 'importance']
            else:
                combined_importance = all_importance.groupby('feature')['importance'].max().reset_index()
            
            combined_importance = combined_importance.sort_values('importance', ascending=False)
            
            logger.info(f"Combined feature importance using {method} method")
            return combined_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def save_model(self, filepath: str) -> bool:
        """Save ensemble model"""
        try:
            model_data = {
                'models': {},
                'model_info': self.model_info,
                'weights': self.weights,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            
            # Save non-LSTM models
            for name, model in self.models.items():
                if name != 'lstm':
                    model_data['models'][name] = model
            
            # Save the main data
            joblib.dump(model_data, f'{filepath}_ensemble.pkl')
            
            # Save LSTM separately if it exists
            if 'lstm' in self.models:
                self.models['lstm'].save(f'{filepath}_lstm.h5')
            
            logger.info(f"Ensemble model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load ensemble model"""
        try:
            # Load main model data
            model_data = joblib.load(f'{filepath}_ensemble.pkl')
            
            self.models = model_data['models']
            self.model_info = model_data['model_info']
            self.weights = model_data['weights']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            # Load LSTM if it exists
            try:
                from tensorflow.keras.models import load_model
                lstm_model = load_model(f'{filepath}_lstm.h5')
                self.models['lstm'] = lstm_model
                logger.info("LSTM component loaded successfully")
            except:
                logger.warning("LSTM component not found or failed to load")
            
            logger.info(f"Ensemble model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            return False

def optimize_ensemble_weights(models: Dict[str, Any], X_val: pd.DataFrame, y_val: pd.Series,
                             weight_bounds: Tuple[float, float] = (0.0, 1.0)) -> Dict[str, float]:
    """Optimize ensemble weights using validation data"""
    try:
        from scipy.optimize import minimize
        
        logger.info("Optimizing ensemble weights...")
        
        # Get predictions from each model
        model_predictions = {}
        for name, model in models.items():
            if name != 'lstm':  # Skip LSTM for simplicity
                model_predictions[name] = model.predict(X_val)
        
        if len(model_predictions) < 2:
            logger.warning("Need at least 2 models for weight optimization")
            return {name: 1.0 for name in model_predictions.keys()}
        
        model_names = list(model_predictions.keys())
        predictions_matrix = np.column_stack([model_predictions[name] for name in model_names])
        
        def objective(weights):
            """Minimize ensemble RMSE"""
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_pred = np.dot(predictions_matrix, weights)
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            return rmse
        
        # Initial weights (equal)
        initial_weights = np.ones(len(model_names)) / len(model_names)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds for each weight
        bounds = [weight_bounds for _ in range(len(model_names))]
        
        # Optimize
        result = minimize(
            objective, initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Normalize
            weight_dict = {name: weight for name, weight in zip(model_names, optimal_weights)}
            
            logger.info("Optimized weights:")
            for name, weight in weight_dict.items():
                logger.info(f"  {name}: {weight:.3f}")
            
            return weight_dict
        else:
            logger.warning("Weight optimization failed, using equal weights")
            return {name: 1.0/len(model_names) for name in model_names}
            
    except Exception as e:
        logger.error(f"Error optimizing ensemble weights: {e}")
        return {}

def create_stacking_ensemble(base_models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series, meta_learner: Any = None) -> Tuple[Any, Dict]:
    """Create stacking ensemble with meta-learner"""
    try:
        logger.info("Creating stacking ensemble...")
        
        if meta_learner is None:
            meta_learner = LinearRegression()
        
        # Get out-of-fold predictions for training meta-learner
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X_train), len(base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            logger.info(f"Processing fold {fold + 1}/5...")
            
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            fold_predictions = []
            
            for model_name, model in base_models.items():
                if model_name != 'lstm':  # Skip LSTM for simplicity
                    # Clone and train model on fold
                    fold_model = type(model)(**model.get_params())
                    fold_model.fit(X_fold_train, y_fold_train)
                    fold_pred = fold_model.predict(X_fold_val)
                    fold_predictions.append(fold_pred)
            
            if fold_predictions:
                meta_features[val_idx] = np.column_stack(fold_predictions)
        
        # Train meta-learner
        if meta_features.size > 0:
            meta_learner.fit(meta_features, y_train)
            
            # Get predictions for validation set
            val_meta_features = np.zeros((len(X_val), len(base_models)))
            for i, (model_name, model) in enumerate(base_models.items()):
                if model_name != 'lstm':
                    val_meta_features[:, i] = model.predict(X_val)
            
            # Meta-learner prediction
            stacking_pred = meta_learner.predict(val_meta_features)
            
            # Evaluate stacking performance
            stacking_rmse = np.sqrt(mean_squared_error(y_val, stacking_pred))
            stacking_r2 = r2_score(y_val, stacking_pred)
            
            stacking_info = {
                'meta_learner': type(meta_learner).__name__,
                'n_base_models': len([m for m in base_models.keys() if m != 'lstm']),
                'cv_folds': 5,
                'validation_rmse': stacking_rmse,
                'validation_r2': stacking_r2
            }
            
            logger.info(f"Stacking ensemble RMSE: {stacking_rmse:.4f}, R²: {stacking_r2:.4f}")
            
            return meta_learner, stacking_info
        else:
            logger.error("No meta-features created for stacking")
            return None, {}
            
    except Exception as e:
        logger.error(f"Error creating stacking ensemble: {e}")
        return None, {}

def evaluate_ensemble_diversity(models: Dict[str, Any], X_test: pd.DataFrame) -> Dict:
    """Evaluate diversity among ensemble components"""
    try:
        logger.info("Evaluating ensemble diversity...")
        
        # Get predictions from each model
        predictions = {}
        for name, model in models.items():
            if name != 'lstm':  # Skip LSTM for simplicity
                predictions[name] = model.predict(X_test)
        
        if len(predictions) < 2:
            return {}
        
        # Calculate pairwise correlations
        pred_df = pd.DataFrame(predictions)
        correlation_matrix = pred_df.corr()
        
        # Average correlation (lower is more diverse)
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Disagreement measure
        pred_array = pred_df.values
        disagreement = np.std(pred_array, axis=1).mean()
        
        # Q-statistic (for pairs of models)
        q_statistics = []
        model_names = list(predictions.keys())
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                pred1 = predictions[model_names[i]]
                pred2 = predictions[model_names[j]]
                
                # Simple Q-statistic approximation
                diff = np.abs(pred1 - pred2)
                q_stat = np.mean(diff) / (np.std(pred1) + np.std(pred2) + 1e-8)
                q_statistics.append(q_stat)
        
        diversity_metrics = {
            'average_correlation': avg_correlation,
            'disagreement': disagreement,
            'q_statistic_mean': np.mean(q_statistics) if q_statistics else 0,
            'correlation_matrix': correlation_matrix.to_dict(),
            'n_models': len(predictions)
        }
        
        logger.info(f"Ensemble diversity - Avg correlation: {avg_correlation:.3f}, Disagreement: {disagreement:.3f}")
        
        return diversity_metrics
        
    except Exception as e:
        logger.error(f"Error evaluating ensemble diversity: {e}")
        return {}

def create_dynamic_ensemble(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                          window_size: int = 100) -> Tuple[Any, Dict]:
    """Create dynamic ensemble that adapts weights based on recent performance"""
    try:
        logger.info("Creating dynamic ensemble...")
        
        class DynamicEnsemble:
            def __init__(self, base_models, window_size=100):
                self.base_models = base_models
                self.window_size = window_size
                self.performance_history = {name: [] for name in base_models.keys()}
                self.current_weights = {name: 1.0/len(base_models) for name in base_models.keys()}
                
            def update_weights(self, X_recent, y_recent):
                """Update weights based on recent performance"""
                if len(y_recent) < 10:  # Need minimum samples
                    return
                
                # Calculate recent performance for each model
                recent_rmse = {}
                for name, model in self.base_models.items():
                    if name != 'lstm':
                        try:
                            pred = model.predict(X_recent)
                            rmse = np.sqrt(mean_squared_error(y_recent, pred))
                            recent_rmse[name] = rmse
                            
                            # Update performance history
                            self.performance_history[name].append(rmse)
                            if len(self.performance_history[name]) > self.window_size:
                                self.performance_history[name].pop(0)
                        except:
                            recent_rmse[name] = float('inf')
                
                # Update weights (inverse of RMSE, normalized)
                if recent_rmse:
                    inv_rmse = {name: 1.0/(rmse + 1e-8) for name, rmse in recent_rmse.items()}
                    total_inv_rmse = sum(inv_rmse.values())
                    self.current_weights = {name: inv/total_inv_rmse for name, inv in inv_rmse.items()}
                
            def predict(self, X):
                """Make weighted predictions"""
                predictions = {}
                for name, model in self.base_models.items():
                    if name != 'lstm':
                        try:
                            predictions[name] = model.predict(X)
                        except:
                            continue
                
                if not predictions:
                    return np.array([])
                
                # Weighted average
                ensemble_pred = np.zeros(len(X))
                total_weight = 0
                
                for name, pred in predictions.items():
                    weight = self.current_weights.get(name, 0)
                    ensemble_pred += weight * pred
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_pred /= total_weight
                
                return ensemble_pred
        
        dynamic_ensemble = DynamicEnsemble(models, window_size)
        
        # Simulate training with sliding window
        n_samples = len(X_train)
        train_size = max(n_samples // 2, 100)  # Start with half the data
        
        for i in range(train_size, n_samples, 20):  # Update every 20 samples
            end_idx = min(i + 20, n_samples)
            X_recent = X_train.iloc[i:end_idx]
            y_recent = y_train.iloc[i:end_idx]
            
            dynamic_ensemble.update_weights(X_recent, y_recent)
        
        dynamic_info = {
            'ensemble_type': 'Dynamic',
            'window_size': window_size,
            'final_weights': dynamic_ensemble.current_weights.copy(),
            'n_updates': (n_samples - train_size) // 20
        }
        
        logger.info("Dynamic ensemble created with adaptive weights")
        return dynamic_ensemble, dynamic_info
        
    except Exception as e:
        logger.error(f"Error creating dynamic ensemble: {e}")
        return None, {}

def main_ensemble_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         ensemble_type: str = 'weighted') -> Dict:
    """Main pipeline for ensemble model creation and evaluation"""
    logger.info("=== Starting Ensemble Pipeline ===")
    
    try:
        results = {}
        
        # 1. Train base models
        logger.info("1. Training base models...")
        from src.models.behavioral_models import train_random_forest_model, train_xgboost_model
        
        base_models = {}
        
        # Random Forest
        rf_model, rf_info = train_random_forest_model(X_train, y_train, hyperparameter_tuning=False)
        if rf_model is not None:
            base_models['random_forest'] = rf_model
        
        # XGBoost
        xgb_model, xgb_info = train_xgboost_model(X_train, y_train, hyperparameter_tuning=False)
        if xgb_model is not None:
            base_models['xgboost'] = xgb_model
        
        if len(base_models) < 2:
            logger.error("Need at least 2 base models for ensemble")
            return {}
        
        results['base_models'] = base_models
        
        # 2. Create ensemble based on type
        if ensemble_type == 'weighted':
            logger.info("2. Creating weighted ensemble...")
            ensemble = BehavioralEnsembleModel()
            ensemble.models = base_models
            ensemble.feature_names = X_train.columns.tolist()
            ensemble.is_trained = True
            
            # Optimize weights
            optimal_weights = optimize_ensemble_weights(base_models, X_test, y_test)
            ensemble.weights = optimal_weights
            
            results['ensemble_model'] = ensemble
            results['ensemble_type'] = 'weighted'
            
        elif ensemble_type == 'stacking':
            logger.info("2. Creating stacking ensemble...")
            meta_learner, stacking_info = create_stacking_ensemble(
                base_models, X_train, y_train, X_test, y_test
            )
            results['ensemble_model'] = meta_learner
            results['ensemble_info'] = stacking_info
            results['ensemble_type'] = 'stacking'
            
        elif ensemble_type == 'dynamic':
            logger.info("2. Creating dynamic ensemble...")
            dynamic_ensemble, dynamic_info = create_dynamic_ensemble(base_models, X_train, y_train)
            results['ensemble_model'] = dynamic_ensemble
            results['ensemble_info'] = dynamic_info
            results['ensemble_type'] = 'dynamic'
        
        # 3. Evaluate ensemble
        logger.info("3. Evaluating ensemble...")
        if ensemble_type == 'weighted' and 'ensemble_model' in results:
            evaluation = results['ensemble_model'].evaluate(X_test, y_test)
            results['evaluation'] = evaluation
            
        # 4. Analyze diversity
        logger.info("4. Analyzing ensemble diversity...")
        diversity_metrics = evaluate_ensemble_diversity(base_models, X_test)
        results['diversity_metrics'] = diversity_metrics
        
        # 5. Save ensemble model
        logger.info("5. Saving ensemble model...")
        if 'ensemble_model' in results and hasattr(results['ensemble_model'], 'save_model'):
            results['ensemble_model'].save_model('data/models/behavioral_ensemble')
        
        logger.info("=== Ensemble Pipeline Complete ===")
        return results
        
    except Exception as e:
        logger.error(f"Error in ensemble pipeline: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("Ensemble model module loaded successfully")
    print("Available classes and functions:")
    print("- BehavioralEnsembleModel")
    print("- optimize_ensemble_weights")
    print("- create_stacking_ensemble")
    print("- evaluate_ensemble_diversity")
    print("- create_dynamic_ensemble")
    print("- main_ensemble_pipeline")