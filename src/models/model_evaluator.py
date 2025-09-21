import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               sample_weight: np.ndarray = None) -> Dict[str, float]:
    """Calculate comprehensive regression metrics"""
    try:
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['r2_score'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
        
        # MAPE with protection against division by zero
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.inf
        
        # Directional accuracy
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
            metrics['directional_accuracy'] = directional_accuracy
        else:
            metrics['directional_accuracy'] = 0.0
        
        # Additional metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        metrics['median_abs_error'] = np.median(np.abs(residuals))
        
        # Percentage of predictions within tolerance
        tolerance_levels = [0.05, 0.1, 0.2]  # 5%, 10%, 20%
        for tol in tolerance_levels:
            within_tolerance = np.mean(np.abs(residuals) <= tol * np.abs(y_true)) * 100
            metrics[f'within_{int(tol*100)}pct_tolerance'] = within_tolerance
        
        logger.info(f"Calculated regression metrics - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2_score']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}")
        return {}

def perform_cross_validation(model: Any, X: pd.DataFrame, y: pd.Series, 
                           cv_method: str = 'timeseries', n_splits: int = 5) -> Dict[str, Any]:
    """Perform cross-validation with different methods"""
    try:
        cv_results = {}
        
        if cv_method == 'timeseries':
            # Time series split for temporal data
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_iterator = tscv
        else:
            # Standard k-fold
            cv_iterator = n_splits
        
        # Perform cross-validation for different metrics
        scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv_iterator, scoring=metric, n_jobs=-1)
            
            # Convert negative scores back to positive for MSE and MAE
            if 'neg_' in metric:
                scores = -scores
                metric_name = metric.replace('neg_', '')
            else:
                metric_name = metric
            
            cv_results[f'{metric_name}_scores'] = scores
            cv_results[f'{metric_name}_mean'] = np.mean(scores)
            cv_results[f'{metric_name}_std'] = np.std(scores)
            cv_results[f'{metric_name}_min'] = np.min(scores)
            cv_results[f'{metric_name}_max'] = np.max(scores)
        
        # RMSE from MSE
        if 'mean_squared_error_scores' in cv_results:
            rmse_scores = np.sqrt(cv_results['mean_squared_error_scores'])
            cv_results['rmse_scores'] = rmse_scores
            cv_results['rmse_mean'] = np.mean(rmse_scores)
            cv_results['rmse_std'] = np.std(rmse_scores)
        
        cv_results['n_splits'] = n_splits
        cv_results['cv_method'] = cv_method
        
        logger.info(f"Cross-validation completed - {cv_method} with {n_splits} splits")
        return cv_results
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        return {}

def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Analyze model residuals for diagnostic purposes"""
    try:
        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        analysis = {
            'residuals_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'median': np.median(residuals),
                'q25': np.percentile(residuals, 25),
                'q75': np.percentile(residuals, 75)
            },
            'normality_tests': {},
            'autocorrelation': {},
            'heteroscedasticity': {}
        }
        
        # Normality tests
        from scipy import stats
        
        # Shapiro-Wilk test (for smaller samples)
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            analysis['normality_tests']['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(residuals)
        analysis['normality_tests']['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > 0.05
        }
        
        # Skewness and kurtosis
        analysis['normality_tests']['skewness'] = stats.skew(residuals)
        analysis['normality_tests']['kurtosis'] = stats.kurtosis(residuals)
        
        # Autocorrelation (Ljung-Box test approximation)
        if len(residuals) > 10:
            # Simple lag-1 autocorrelation
            lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            analysis['autocorrelation']['lag_1'] = lag1_corr
            analysis['autocorrelation']['significant'] = abs(lag1_corr) > 0.1
        
        # Heteroscedasticity (Breusch-Pagan test approximation)
        # Split residuals into groups and compare variances
        n_groups = 3
        group_size = len(residuals) // n_groups
        groups = [residuals[i*group_size:(i+1)*group_size] for i in range(n_groups)]
        
        if all(len(group) > 1 for group in groups):
            group_vars = [np.var(group) for group in groups]
            var_ratio = max(group_vars) / min(group_vars)
            analysis['heteroscedasticity']['variance_ratio'] = var_ratio
            analysis['heteroscedasticity']['likely_heteroscedastic'] = var_ratio > 2.0
        
        # Outlier detection
        outlier_threshold = 2.5  # Standard deviations
        outliers = np.abs(standardized_residuals) > outlier_threshold
        analysis['outliers'] = {
            'count': np.sum(outliers),
            'percentage': np.mean(outliers) * 100,
            'threshold': outlier_threshold,
            'indices': np.where(outliers)[0].tolist()
        }
        
        logger.info(f"Residual analysis completed - Mean: {analysis['residuals_stats']['mean']:.6f}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing residuals: {e}")
        return {}

def evaluate_model_stability(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                           n_bootstrap: int = 100, sample_ratio: float = 0.8) -> Dict[str, Any]:
    """Evaluate model stability using bootstrap sampling"""
    try:
        logger.info(f"Evaluating model stability with {n_bootstrap} bootstrap samples")
        
        bootstrap_metrics = {
            'rmse': [], 'mae': [], 'r2': [], 'feature_importance': []
        }
        
        n_samples = int(len(X_train) * sample_ratio)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(X_train), size=n_samples, replace=True)
            X_bootstrap = X_train.iloc[bootstrap_indices]
            y_bootstrap = y_train.iloc[bootstrap_indices]
            
            # Train model on bootstrap sample
            bootstrap_model = type(model)(**model.get_params())
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Predict on original training data
            y_pred_bootstrap = bootstrap_model.predict(X_train)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_bootstrap))
            mae = mean_absolute_error(y_train, y_pred_bootstrap)
            r2 = r2_score(y_train, y_pred_bootstrap)
            
            bootstrap_metrics['rmse'].append(rmse)
            bootstrap_metrics['mae'].append(mae)
            bootstrap_metrics['r2'].append(r2)
            
            # Feature importance (if available)
            if hasattr(bootstrap_model, 'feature_importances_'):
                bootstrap_metrics['feature_importance'].append(bootstrap_model.feature_importances_)
        
        # Calculate stability statistics
        stability_stats = {}
        
        for metric in ['rmse', 'mae', 'r2']:
            values = bootstrap_metrics[metric]
            stability_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values),  # Coefficient of variation
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        
        # Feature importance stability
        if bootstrap_metrics['feature_importance']:
            feature_importances = np.array(bootstrap_metrics['feature_importance'])
            importance_mean = np.mean(feature_importances, axis=0)
            importance_std = np.std(feature_importances, axis=0)
            importance_cv = importance_std / (importance_mean + 1e-8)
            
            stability_stats['feature_importance'] = {
                'mean_importance': importance_mean,
                'std_importance': importance_std,
                'cv_importance': importance_cv,
                'stable_features': np.sum(importance_cv < 0.5)  # Features with CV < 0.5
            }
        
        stability_stats['n_bootstrap'] = n_bootstrap
        stability_stats['sample_ratio'] = sample_ratio
        
        logger.info(f"Model stability evaluation completed")
        return stability_stats
        
    except Exception as e:
        logger.error(f"Error evaluating model stability: {e}")
        return {}

def compare_models(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compare multiple models on the same test set"""
    try:
        logger.info(f"Comparing {len(models_dict)} models")
        
        comparison_results = []
        
        for model_name, model in models_dict.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = calculate_regression_metrics(y_test.values, y_pred)
                
                # Add model name and type
                metrics['model_name'] = model_name
                metrics['model_type'] = type(model).__name__
                
                # Model complexity (if available)
                if hasattr(model, 'n_estimators'):
                    metrics['n_estimators'] = model.n_estimators
                if hasattr(model, 'max_depth'):
                    metrics['max_depth'] = model.max_depth
                if hasattr(model, 'n_features_in_'):
                    metrics['n_features'] = model.n_features_in_
                
                comparison_results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error evaluating model {model_name}: {e}")
                continue
        
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            
            # Sort by RMSE (best first)
            comparison_df = comparison_df.sort_values('rmse')
            
            logger.info(f"Model comparison completed for {len(comparison_df)} models")
            return comparison_df
        else:
            logger.warning("No models could be evaluated")
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return pd.DataFrame()

def create_evaluation_report(model_name: str, metrics: Dict, cv_results: Dict = None,
                           residual_analysis: Dict = None, stability_analysis: Dict = None) -> Dict:
    """Create comprehensive model evaluation report"""
    try:
        report = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': metrics,
            'model_quality': {}
        }
        
        # Performance assessment
        rmse = metrics.get('rmse', 0)
        r2 = metrics.get('r2_score', 0)
        mape = metrics.get('mape', 0)
        
        # Quality scoring (0-100)
        quality_score = 0
        
        # R² contribution (40%)
        r2_score_contrib = min(40, r2 * 40) if r2 >= 0 else 0
        quality_score += r2_score_contrib
        
        # MAPE contribution (30%)
        if mape < 5:
            mape_contrib = 30
        elif mape < 10:
            mape_contrib = 25
        elif mape < 20:
            mape_contrib = 15
        elif mape < 30:
            mape_contrib = 5
        else:
            mape_contrib = 0
        quality_score += mape_contrib
        
        # Directional accuracy contribution (30%)
        dir_acc = metrics.get('directional_accuracy', 0)
        dir_acc_contrib = min(30, dir_acc * 30 / 100)
        quality_score += dir_acc_contrib
        
        report['model_quality']['overall_score'] = round(quality_score, 1)
        
        # Quality assessment
        if quality_score >= 80:
            quality_rating = "Excellent"
        elif quality_score >= 70:
            quality_rating = "Good"
        elif quality_score >= 60:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        report['model_quality']['rating'] = quality_rating
        
        # Add additional analyses if provided
        if cv_results:
            report['cross_validation'] = cv_results
        
        if residual_analysis:
            report['residual_analysis'] = residual_analysis
            
            # Check for issues
            issues = []
            
            # Normality issues
            jb_test = residual_analysis.get('normality_tests', {}).get('jarque_bera', {})
            if not jb_test.get('is_normal', True):
                issues.append("Residuals not normally distributed")
            
            # Autocorrelation issues
            if residual_analysis.get('autocorrelation', {}).get('significant', False):
                issues.append("Significant autocorrelation in residuals")
            
            # Heteroscedasticity issues
            if residual_analysis.get('heteroscedasticity', {}).get('likely_heteroscedastic', False):
                issues.append("Likely heteroscedasticity present")
            
            # Outliers
            outlier_pct = residual_analysis.get('outliers', {}).get('percentage', 0)
            if outlier_pct > 5:
                issues.append(f"High percentage of outliers ({outlier_pct:.1f}%)")
            
            report['model_quality']['issues'] = issues
        
        if stability_analysis:
            report['stability_analysis'] = stability_analysis
            
            # Stability assessment
            rmse_cv = stability_analysis.get('rmse', {}).get('cv', 0)
            if rmse_cv < 0.1:
                stability_rating = "Very Stable"
            elif rmse_cv < 0.2:
                stability_rating = "Stable"
            elif rmse_cv < 0.3:
                stability_rating = "Moderately Stable"
            else:
                stability_rating = "Unstable"
            
            report['model_quality']['stability'] = stability_rating
        
        # Recommendations
        recommendations = []
        
        if r2 < 0.5:
            recommendations.append("Consider feature engineering or different algorithms")
        
        if mape > 20:
            recommendations.append("High prediction errors - review model assumptions")
        
        if quality_score < 60:
            recommendations.append("Model performance below acceptable threshold")
        
        report['recommendations'] = recommendations
        
        logger.info(f"Evaluation report created for {model_name} - Quality: {quality_rating}")
        return report
        
    except Exception as e:
        logger.error(f"Error creating evaluation report: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("Model evaluator module loaded successfully")
    print("Available functions:")
    print("- calculate_regression_metrics")
    print("- perform_cross_validation")
    print("- analyze_residuals")
    print("- evaluate_model_stability")
    print("- compare_models")
    print("- create_evaluation_report")