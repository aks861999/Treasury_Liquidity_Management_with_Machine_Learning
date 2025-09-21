# Fix the performance_metrics.py file - complete the missing parts

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_prediction_accuracy_metrics(actual: np.ndarray, predicted: np.ndarray, 
                                        dates: np.ndarray = None) -> Dict:
    """Calculate comprehensive prediction accuracy metrics"""
    try:
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_size': len(actual)
        }
        
        # Basic accuracy metrics
        metrics['mse'] = mean_squared_error(actual, predicted)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(actual, predicted)
        metrics['r2'] = r2_score(actual, predicted)
        
        # MAPE with zero protection
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.inf
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = np.diff(actual)
            pred_direction = np.diff(predicted)
            directional_accuracy = np.mean(np.sign(actual_direction) == np.sign(pred_direction)) * 100
            metrics['directional_accuracy'] = directional_accuracy
        
        # Hit rate (percentage of predictions within tolerance)
        tolerance_levels = [0.05, 0.10, 0.20]  # 5%, 10%, 20%
        for tolerance in tolerance_levels:
            within_tolerance = np.mean(np.abs(actual - predicted) <= tolerance * np.abs(actual)) * 100
            metrics[f'hit_rate_{int(tolerance*100)}pct'] = within_tolerance
        
        # Bias metrics
        bias = np.mean(predicted - actual)
        metrics['bias'] = bias
        metrics['bias_percentage'] = (bias / np.mean(actual)) * 100 if np.mean(actual) != 0 else 0
        
        # Volatility metrics
        actual_vol = np.std(actual)
        pred_vol = np.std(predicted)
        metrics['actual_volatility'] = actual_vol
        metrics['predicted_volatility'] = pred_vol
        metrics['volatility_ratio'] = pred_vol / actual_vol if actual_vol != 0 else 0
        
        # Correlation
        correlation = np.corrcoef(actual, predicted)[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0
        
        # Time-based metrics if dates provided
        if dates is not None and len(dates) == len(actual):
            time_metrics = calculate_time_based_metrics(actual, predicted, dates)
            metrics.update(time_metrics)
        
        logger.info(f"Calculated prediction accuracy metrics - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy metrics: {e}")
        return {}

def calculate_time_based_metrics(actual: np.ndarray, predicted: np.ndarray, 
                                dates: np.ndarray) -> Dict:
    """Calculate time-based performance metrics"""
    try:
        time_metrics = {}
        
        # Convert dates to pandas datetime
        dates_pd = pd.to_datetime(dates)
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'date': dates_pd,
            'actual': actual,
            'predicted': predicted,
            'error': actual - predicted,
            'abs_error': np.abs(actual - predicted)
        })
        
        # Monthly performance
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_metrics = df.groupby('year_month').agg({
            'error': ['mean', 'std'],
            'abs_error': 'mean',
            'actual': 'count'
        }).round(4)
        
        monthly_metrics.columns = ['_'.join(col).strip() for col in monthly_metrics.columns.values]
        time_metrics['monthly_performance'] = monthly_metrics.to_dict('index')
        
        # Quarterly performance
        df['year_quarter'] = df['date'].dt.to_period('Q')
        quarterly_metrics = df.groupby('year_quarter').agg({
            'error': ['mean', 'std'],
            'abs_error': 'mean',
            'actual': 'count'
        }).round(4)
        
        quarterly_metrics.columns = ['_'.join(col).strip() for col in quarterly_metrics.columns.values]
        time_metrics['quarterly_performance'] = quarterly_metrics.to_dict('index')
        
        # Performance trend (improving/deteriorating)
        if len(df) >= 60:  # At least 60 observations
            # Split into first and second half
            mid_point = len(df) // 2
            first_half_mae = df.iloc[:mid_point]['abs_error'].mean()
            second_half_mae = df.iloc[mid_point:]['abs_error'].mean()
            
            improvement = (first_half_mae - second_half_mae) / first_half_mae * 100
            time_metrics['performance_trend'] = {
                'first_half_mae': first_half_mae,
                'second_half_mae': second_half_mae,
                'improvement_percentage': improvement,
                'trend': 'improving' if improvement > 5 else 'deteriorating' if improvement < -5 else 'stable'
            }
        
        return time_metrics
        
    except Exception as e:
        logger.error(f"Error calculating time-based metrics: {e}")
        return {}

def calculate_model_efficiency_metrics(training_time: float, prediction_time: float,
                                     memory_usage: float = None, 
                                     n_features: int = None) -> Dict:
    """Calculate model efficiency and resource usage metrics"""
    try:
        efficiency_metrics = {
            'training_time_seconds': training_time,
            'prediction_time_seconds': prediction_time,
            'predictions_per_second': 1 / prediction_time if prediction_time > 0 else 0
        }
        
        if memory_usage is not None:
            efficiency_metrics['memory_usage_mb'] = memory_usage
            efficiency_metrics['memory_efficiency'] = memory_usage / n_features if n_features else 0
        
        if n_features is not None:
            efficiency_metrics['n_features'] = n_features
            efficiency_metrics['training_time_per_feature'] = training_time / n_features
        
        # Efficiency rating
        if training_time < 10 and prediction_time < 0.001:
            efficiency_rating = "Excellent"
        elif training_time < 60 and prediction_time < 0.01:
            efficiency_rating = "Good"
        elif training_time < 300 and prediction_time < 0.1:
            efficiency_rating = "Fair"
        else:
            efficiency_rating = "Poor"
        
        efficiency_metrics['efficiency_rating'] = efficiency_rating
        
        logger.info(f"Calculated efficiency metrics - Training: {training_time:.2f}s, Prediction: {prediction_time:.4f}s")
        return efficiency_metrics
        
    except Exception as e:
        logger.error(f"Error calculating efficiency metrics: {e}")
        return {}

def calculate_business_impact_metrics(predictions: np.ndarray, actuals: np.ndarray,
                                    business_value_per_unit: float = 1.0,
                                    cost_per_error: float = 1.0) -> Dict:
    """Calculate business impact metrics"""
    try:
        errors = np.abs(predictions - actuals)
        
        business_metrics = {
            'total_prediction_value': np.sum(predictions) * business_value_per_unit,
            'total_actual_value': np.sum(actuals) * business_value_per_unit,
            'value_difference': (np.sum(predictions) - np.sum(actuals)) * business_value_per_unit,
            'total_error_cost': np.sum(errors) * cost_per_error,
            'average_error_cost': np.mean(errors) * cost_per_error,
            'max_error_cost': np.max(errors) * cost_per_error
        }
        
        # Risk-adjusted metrics
        business_metrics['value_at_risk_95'] = np.percentile(errors, 95) * cost_per_error
        business_metrics['expected_shortfall'] = np.mean(errors[errors > np.percentile(errors, 95)]) * cost_per_error
        
        # ROI calculation (simplified)
        potential_savings = business_metrics['total_actual_value'] * 0.01  # Assume 1% savings potential
        model_cost = business_metrics['total_error_cost']
        business_metrics['roi_percentage'] = ((potential_savings - model_cost) / model_cost) * 100 if model_cost > 0 else 0
        
        logger.info(f"Calculated business impact metrics - Total error cost: {business_metrics['total_error_cost']:.2f}")
        return business_metrics
        
    except Exception as e:
        logger.error(f"Error calculating business impact metrics: {e}")
        return {}

def calculate_model_stability_metrics(predictions_list: List[np.ndarray], 
                                    model_names: List[str] = None) -> Dict:
    """Calculate stability metrics across multiple model runs"""
    try:
        if not predictions_list:
            return {}
        
        # Stack predictions
        predictions_array = np.array(predictions_list)
        
        stability_metrics = {
            'n_runs': len(predictions_list),
            'mean_prediction': np.mean(predictions_array, axis=0),
            'prediction_std': np.std(predictions_array, axis=0),
            'prediction_range': np.max(predictions_array, axis=0) - np.min(predictions_array, axis=0)
        }
        
        # Overall stability measures
        stability_metrics['average_std'] = np.mean(stability_metrics['prediction_std'])
        stability_metrics['max_std'] = np.max(stability_metrics['prediction_std'])
        stability_metrics['coefficient_of_variation'] = stability_metrics['average_std'] / np.mean(stability_metrics['mean_prediction']) if np.mean(stability_metrics['mean_prediction']) != 0 else 0
        
        # Stability rating
        cv = stability_metrics['coefficient_of_variation']
        if cv < 0.05:
            stability_rating = "Very Stable"
        elif cv < 0.1:
            stability_rating = "Stable"
        elif cv < 0.2:
            stability_rating = "Moderately Stable"
        else:
            stability_rating = "Unstable"
        
        stability_metrics['stability_rating'] = stability_rating
        
        # Pairwise correlations between runs
        if len(predictions_list) > 1:
            correlations = []
            for i in range(len(predictions_list)):
                for j in range(i+1, len(predictions_list)):
                    corr = np.corrcoef(predictions_list[i], predictions_list[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                stability_metrics['inter_run_correlation'] = {
                    'mean': np.mean(correlations),
                    'min': np.min(correlations),
                    'max': np.max(correlations),
                    'std': np.std(correlations)
                }
        
        logger.info(f"Calculated stability metrics for {len(predictions_list)} runs - Rating: {stability_rating}")
        return stability_metrics
        
    except Exception as e:
        logger.error(f"Error calculating stability metrics: {e}")
        return {}

def calculate_feature_importance_metrics(feature_importances: np.ndarray, 
                                       feature_names: List[str] = None) -> Dict:
    """Calculate feature importance analysis metrics"""
    try:
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(feature_importances))]
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        importance_metrics = {
            'n_features': len(feature_importances),
            'total_importance': np.sum(feature_importances),
            'mean_importance': np.mean(feature_importances),
            'std_importance': np.std(feature_importances),
            'max_importance': np.max(feature_importances),
            'min_importance': np.min(feature_importances)
        }
        
        # Concentration metrics
        cumsum_importance = np.cumsum(sorted_importances) / np.sum(sorted_importances)
        
        # Top N feature contributions
        for n in [5, 10, 20]:
            if n <= len(feature_importances):
                top_n_contribution = cumsum_importance[n-1] * 100
                importance_metrics[f'top_{n}_contribution_pct'] = top_n_contribution
        
        # Gini coefficient for importance concentration
        sorted_importances_norm = sorted_importances / np.sum(sorted_importances)
        n = len(sorted_importances_norm)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_importances_norm)) / (n * np.sum(sorted_importances_norm))
        importance_metrics['gini_coefficient'] = gini
        
        # Feature importance distribution
        importance_metrics['feature_ranking'] = {
            'names': sorted_names[:20],  # Top 20
            'importances': sorted_importances[:20].tolist(),
            'cumulative_importance': (cumsum_importance[:20] * 100).tolist()
        }
        
        # Diversity metrics
        non_zero_features = np.sum(feature_importances > 0.001)  # Features with >0.1% importance
        importance_metrics['effective_features'] = non_zero_features
        importance_metrics['feature_diversity'] = non_zero_features / len(feature_importances)
        
        logger.info(f"Calculated feature importance metrics - Top feature contributes {importance_metrics.get('top_5_contribution_pct', 0):.1f}%")
        return importance_metrics
        
    except Exception as e:
        logger.error(f"Error calculating feature importance metrics: {e}")
        return {}

def calculate_comparative_performance(model_results: Dict[str, Dict]) -> Dict:
    """Calculate comparative performance metrics across multiple models"""
    try:
        if not model_results:
            return {}
        
        model_names = list(model_results.keys())
        
        comparative_metrics = {
            'n_models': len(model_names),
            'model_names': model_names,
            'metric_comparison': {}
        }
        
        # Extract common metrics
        common_metrics = set()
        for results in model_results.values():
            if isinstance(results, dict):
                common_metrics.update(results.keys())
        
        # Compare each metric
        for metric in common_metrics:
            if metric in ['rmse', 'mae', 'mape', 'r2', 'directional_accuracy']:
                values = []
                for model_name in model_names:
                    if metric in model_results[model_name]:
                        values.append(model_results[model_name][metric])
                
                if values:
                    comparative_metrics['metric_comparison'][metric] = {
                        'values': dict(zip(model_names, values)),
                        'best_model': model_names[np.argmin(values) if metric in ['rmse', 'mae', 'mape'] else np.argmax(values)],
                        'worst_model': model_names[np.argmax(values) if metric in ['rmse', 'mae', 'mape'] else np.argmin(values)],
                        'spread': max(values) - min(values),
                        'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    }
        
        # Overall ranking
        ranking_metrics = ['rmse', 'r2', 'directional_accuracy']
        available_ranking_metrics = [m for m in ranking_metrics if m in comparative_metrics['metric_comparison']]
        
        if available_ranking_metrics:
            model_scores = {}
            
            for model_name in model_names:
                score = 0
                for metric in available_ranking_metrics:
                    values = list(comparative_metrics['metric_comparison'][metric]['values'].values())
                    model_value = comparative_metrics['metric_comparison'][metric]['values'][model_name]
                    
                    # Normalize and score
                    if metric in ['rmse', 'mae', 'mape']:
                        # Lower is better
                        normalized_score = 1 - (model_value - min(values)) / (max(values) - min(values) + 1e-8)
                    else:
                        # Higher is better
                        normalized_score = (model_value - min(values)) / (max(values) - min(values) + 1e-8)
                    
                    score += normalized_score
                
                model_scores[model_name] = score / len(available_ranking_metrics)
            
            # Sort by score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            comparative_metrics['overall_ranking'] = {
                'rankings': [{'model': model, 'score': score} for model, score in sorted_models],
                'best_model': sorted_models[0][0],
                'worst_model': sorted_models[-1][0]
            }
        
        logger.info(f"Calculated comparative performance for {len(model_names)} models")
        return comparative_metrics
        
    except Exception as e:
        logger.error(f"Error calculating comparative performance: {e}")
        return {}

def generate_performance_report(model_name: str, accuracy_metrics: Dict,
                              efficiency_metrics: Dict = None, 
                              business_metrics: Dict = None,
                              stability_metrics: Dict = None) -> Dict:
    """Generate comprehensive performance report"""
    try:
        report = {
            'model_name': model_name,
            'report_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'executive_summary': {},
            'detailed_metrics': {
                'accuracy': accuracy_metrics
            },
            'performance_rating': {},
            'recommendations': []
        }
        
        # Add additional metrics if provided
        if efficiency_metrics:
            report['detailed_metrics']['efficiency'] = efficiency_metrics
        if business_metrics:
            report['detailed_metrics']['business_impact'] = business_metrics
        if stability_metrics:
            report['detailed_metrics']['stability'] = stability_metrics
        
        # Executive summary
        rmse = accuracy_metrics.get('rmse', 0)
        r2 = accuracy_metrics.get('r2', 0)
        mape = accuracy_metrics.get('mape', 0)
        
        report['executive_summary'] = {
            'prediction_accuracy': f"RMSE: {rmse:.4f}, R²: {r2:.4f}",
            'error_rate': f"MAPE: {mape:.2f}%" if mape != np.inf else "MAPE: N/A",
            'sample_size': accuracy_metrics.get('sample_size', 0)
        }
        
        # Performance rating
        rating_score = 0
        
        # Accuracy rating (50%)
        if r2 >= 0.8:
            accuracy_rating = "Excellent"
            rating_score += 50
        elif r2 >= 0.6:
            accuracy_rating = "Good"
            rating_score += 37.5
        elif r2 >= 0.4:
            accuracy_rating = "Fair"
            rating_score += 25
        else:
            accuracy_rating = "Poor"
            rating_score += 12.5
        
        # Efficiency rating (20%)
        if efficiency_metrics:
            eff_rating = efficiency_metrics.get('efficiency_rating', 'Fair')
            if eff_rating == "Excellent":
                rating_score += 20
            elif eff_rating == "Good":
                rating_score += 15
            elif eff_rating == "Fair":
                rating_score += 10
            else:
                rating_score += 5
        
        # Stability rating (20%)
        if stability_metrics:
            stab_rating = stability_metrics.get('stability_rating', 'Moderately Stable')
            if stab_rating == "Very Stable":
                rating_score += 20
            elif stab_rating == "Stable":
                rating_score += 15
            elif stab_rating == "Moderately Stable":
                rating_score += 10
            else:
                rating_score += 5
        
        # Business impact rating (10%)
        if business_metrics:
            roi = business_metrics.get('roi_percentage', 0)
            if roi > 20:
                rating_score += 10
            elif roi > 0:
                rating_score += 7.5
            elif roi > -10:
                rating_score += 5
            else:
                rating_score += 2.5
        
        # Overall rating
        if rating_score >= 85:
            overall_rating = "Excellent"
        elif rating_score >= 70:
            overall_rating = "Good"
        elif rating_score >= 55:
            overall_rating = "Fair"
        else:
            overall_rating = "Poor"
        
        report['performance_rating'] = {
            'accuracy_rating': accuracy_rating,
            'overall_rating': overall_rating,
            'overall_score': round(rating_score, 1)
        }
        
        # Recommendations
        recommendations = []
        
        if r2 < 0.6:
            recommendations.append("Consider feature engineering or alternative algorithms to improve accuracy")
        
        if mape > 15 and mape != np.inf:
            recommendations.append("High error rate - review model assumptions and data quality")
        
        if efficiency_metrics and efficiency_metrics.get('training_time_seconds', 0) > 300:
            recommendations.append("Consider model simplification to reduce training time")
        
        if stability_metrics and stability_metrics.get('stability_rating') in ['Unstable', 'Moderately Stable']:
            recommendations.append("Model stability issues - consider ensemble methods or regularization")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory - monitor for any degradation")
        
        report['recommendations'] = recommendations
        
        logger.info(f"Generated performance report for {model_name} - Rating: {overall_rating}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("Performance metrics module loaded successfully")
    print("Available functions:")
    print("- calculate_prediction_accuracy_metrics")
    print("- calculate_time_based_metrics")
    print("- calculate_model_efficiency_metrics")
    print("- calculate_business_impact_metrics")
    print("- calculate_model_stability_metrics")
    print("- calculate_feature_importance_metrics")
    print("- calculate_comparative_performance")
    print("- generate_performance_report")