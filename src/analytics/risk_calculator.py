import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional
from config import RISK_CONFIG, VALIDATION_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_lcr_ratio(deposit_data: pd.DataFrame, liquid_assets: float = None) -> Dict:
    """Calculate Liquidity Coverage Ratio (LCR)"""
    try:
        logger.info("Calculating LCR ratio...")
        
        if liquid_assets is None:
            # Estimate liquid assets as percentage of total deposits
            total_deposits = deposit_data['balance'].sum() if 'balance' in deposit_data.columns else 1e9
            liquid_assets = total_deposits * 0.15  # Conservative 15%
        
        # Calculate net cash outflows over 30 days
        # This is simplified - real LCR calculation is more complex
        if 'deposit_flow' in deposit_data.columns:
            daily_outflows = deposit_data[deposit_data['deposit_flow'] < 0]['deposit_flow'].abs()
            avg_daily_outflow = daily_outflows.mean() if not daily_outflows.empty else 0
            net_30day_outflow = avg_daily_outflow * 30
        else:
            net_30day_outflow = liquid_assets * 0.1  # Assume 10% outflow
        
        # LCR calculation
        lcr_ratio = liquid_assets / (net_30day_outflow + 1e-8)
        
        # Risk assessment
        risk_level = "Low" if lcr_ratio >= RISK_CONFIG['LCR_MINIMUM'] + RISK_CONFIG['LCR_BUFFER'] else \
                    "Medium" if lcr_ratio >= RISK_CONFIG['LCR_MINIMUM'] else "High"
        
        lcr_results = {
            'lcr_ratio': lcr_ratio,
            'liquid_assets': liquid_assets,
            'net_30day_outflow': net_30day_outflow,
            'minimum_required': RISK_CONFIG['LCR_MINIMUM'],
            'buffer_target': RISK_CONFIG['LCR_MINIMUM'] + RISK_CONFIG['LCR_BUFFER'],
            'risk_level': risk_level,
            'excess_liquidity': liquid_assets - net_30day_outflow * RISK_CONFIG['LCR_MINIMUM'],
            'days_to_depletion': liquid_assets / (avg_daily_outflow + 1e-8) if 'avg_daily_outflow' in locals() else float('inf')
        }
        
        logger.info(f"LCR ratio calculated: {lcr_ratio:.2%} ({risk_level} risk)")
        return lcr_results
        
    except Exception as e:
        logger.error(f"Error calculating LCR ratio: {e}")
        return {}

def calculate_deposit_at_risk(deposit_flows: pd.DataFrame, confidence_level: float = 0.95) -> Dict:
    """Calculate Deposit at Risk (DaR) using VaR methodology"""
    try:
        logger.info(f"Calculating Deposit at Risk at {confidence_level:.1%} confidence...")
        
        if 'deposit_flow' not in deposit_flows.columns:
            logger.error("Deposit flow column not found")
            return {}
        
        flows = deposit_flows['deposit_flow'].dropna()
        
        if len(flows) < 30:
            logger.warning("Insufficient data for reliable DaR calculation")
            return {}
        
        # Historical simulation VaR
        var_percentile = (1 - confidence_level) * 100
        historical_var = np.percentile(flows, var_percentile)
        
        # Parametric VaR (assuming normal distribution)
        mean_flow = flows.mean()
        std_flow = flows.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        parametric_var = mean_flow + z_score * std_flow
        
        # Expected Shortfall (Conditional VaR)
        tail_flows = flows[flows <= historical_var]
        expected_shortfall = tail_flows.mean() if not tail_flows.empty else historical_var
        
        # Monte Carlo simulation
        n_simulations = 10000
        np.random.seed(42)
        simulated_flows = np.random.normal(mean_flow, std_flow, n_simulations)
        monte_carlo_var = np.percentile(simulated_flows, var_percentile)
        
        dar_results = {
            'confidence_level': confidence_level,
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'monte_carlo_var': monte_carlo_var,
            'expected_shortfall': expected_shortfall,
            'var_ratio': abs(historical_var) / (std_flow + 1e-8),
            'flow_statistics': {
                'mean': mean_flow,
                'std': std_flow,
                'skewness': stats.skew(flows),
                'kurtosis': stats.kurtosis(flows),
                'min': flows.min(),
                'max': flows.max()
            },
            'backtesting_violations': len(flows[flows < historical_var]) / len(flows)
        }
        
        logger.info(f"DaR calculated - Historical VaR: {historical_var:,.0f}")
        return dar_results
        
    except Exception as e:
        logger.error(f"Error calculating Deposit at Risk: {e}")
        return {}

def calculate_concentration_risk(customer_data: pd.DataFrame, deposit_data: pd.DataFrame) -> Dict:
    """Calculate concentration risk metrics"""
    try:
        logger.info("Calculating concentration risk...")
        
        concentration_metrics = {}
        
        # Segment concentration
        if 'segment' in customer_data.columns and 'balance_avg' in customer_data.columns:
            segment_exposure = customer_data.groupby('segment')['balance_avg'].sum()
            total_exposure = segment_exposure.sum()
            
            segment_concentration = segment_exposure / total_exposure
            
            # Herfindahl-Hirschman Index (HHI)
            hhi = (segment_concentration ** 2).sum()
            
            # Concentration ratio (top segments)
            top_3_concentration = segment_concentration.nlargest(3).sum()
            
            concentration_metrics['segment_analysis'] = {
                'hhi_index': hhi,
                'top_3_concentration': top_3_concentration,
                'segment_distribution': segment_concentration.to_dict(),
                'n_segments': len(segment_exposure),
                'max_segment_exposure': segment_concentration.max()
            }
        
        # Geographic concentration (if available)
        if 'region' in customer_data.columns:
            region_exposure = customer_data.groupby('region')['balance_avg'].sum()
            total_exposure = region_exposure.sum()
            region_concentration = region_exposure / total_exposure
            
            concentration_metrics['geographic_analysis'] = {
                'region_distribution': region_concentration.to_dict(),
                'max_region_exposure': region_concentration.max(),
                'n_regions': len(region_exposure)
            }
        
        # Large depositor analysis
        if 'balance_avg' in customer_data.columns:
            balances = customer_data['balance_avg']
            
            # Identify large depositors (top 5%)
            large_depositor_threshold = balances.quantile(0.95)
            large_depositors = customer_data[balances >= large_depositor_threshold]
            
            large_dep_exposure = large_depositors['balance_avg'].sum()
            large_dep_concentration = large_dep_exposure / balances.sum()
            
            concentration_metrics['large_depositor_analysis'] = {
                'threshold': large_depositor_threshold,
                'n_large_depositors': len(large_depositors),
                'large_depositor_concentration': large_dep_concentration,
                'avg_large_depositor_balance': large_depositors['balance_avg'].mean(),
                'largest_depositor_balance': balances.max()
            }
        
        # Risk assessment
        risk_flags = []
        if concentration_metrics.get('segment_analysis', {}).get('top_3_concentration', 0) > 0.7:
            risk_flags.append("High segment concentration")
        
        if concentration_metrics.get('large_depositor_analysis', {}).get('large_depositor_concentration', 0) > 0.3:
            risk_flags.append("High large depositor concentration")
        
        concentration_metrics['risk_assessment'] = {
            'risk_flags': risk_flags,
            'overall_risk': "High" if len(risk_flags) >= 2 else "Medium" if len(risk_flags) == 1 else "Low"
        }
        
        logger.info(f"Concentration risk calculated - Overall risk: {concentration_metrics['risk_assessment']['overall_risk']}")
        return concentration_metrics
        
    except Exception as e:
        logger.error(f"Error calculating concentration risk: {e}")
        return {}

def calculate_behavioral_risk_scores(customer_data: pd.DataFrame, flow_data: pd.DataFrame) -> Dict:
    """Calculate behavioral risk scores for customer segments"""
    try:
        logger.info("Calculating behavioral risk scores...")
        
        behavioral_scores = {}
        
        # Customer-level behavioral metrics
        if all(col in customer_data.columns for col in ['rate_sensitivity', 'loyalty_score', 'volatility_factor']):
            
            # Risk scoring based on behavioral characteristics
            customer_risk_scores = customer_data.copy()
            
            # Normalize behavioral factors (0-1 scale)
            customer_risk_scores['normalized_rate_sensitivity'] = customer_risk_scores['rate_sensitivity']
            customer_risk_scores['normalized_disloyalty'] = 1 - customer_risk_scores['loyalty_score']
            customer_risk_scores['normalized_volatility'] = customer_risk_scores['volatility_factor']
            
            # Weighted risk score
            weights = {'rate_sensitivity': 0.4, 'disloyalty': 0.35, 'volatility': 0.25}
            
            customer_risk_scores['behavioral_risk_score'] = (
                customer_risk_scores['normalized_rate_sensitivity'] * weights['rate_sensitivity'] +
                customer_risk_scores['normalized_disloyalty'] * weights['disloyalty'] +
                customer_risk_scores['normalized_volatility'] * weights['volatility']
            )
            
            # Segment-level aggregation
            segment_risk = customer_risk_scores.groupby('segment').agg({
                'behavioral_risk_score': ['mean', 'std', 'count'],
                'balance_avg': 'sum',
                'rate_sensitivity': 'mean',
                'loyalty_score': 'mean',
                'volatility_factor': 'mean'
            }).round(4)
            
            # Flatten column names
            segment_risk.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in segment_risk.columns]
            
            behavioral_scores['customer_scores'] = customer_risk_scores[
                ['customer_id', 'segment', 'behavioral_risk_score']
            ].to_dict('records')
            
            behavioral_scores['segment_scores'] = segment_risk.to_dict('index')
        
        # Flow-based behavioral analysis
        if 'deposit_flow' in flow_data.columns and 'segment' in flow_data.columns:
            # Flow volatility by segment
            flow_volatility = flow_data.groupby('segment')['deposit_flow'].agg([
                'std', 'mean', 'count', 'min', 'max'
            ])
            
            # Flow persistence (autocorrelation)
            flow_persistence = {}
            for segment in flow_data['segment'].unique():
                segment_flows = flow_data[flow_data['segment'] == segment]['deposit_flow']
                if len(segment_flows) > 10:
                    # Calculate lag-1 autocorrelation
                    persistence = segment_flows.autocorr(lag=1)
                    flow_persistence[segment] = persistence if not np.isnan(persistence) else 0
            
            behavioral_scores['flow_analysis'] = {
                'volatility_by_segment': flow_volatility.to_dict('index'),
                'persistence_by_segment': flow_persistence
            }
        
        # Overall risk ranking
        if 'segment_scores' in behavioral_scores:
            segment_risk_ranking = []
            for segment, scores in behavioral_scores['segment_scores'].items():
                risk_score = scores.get('mean_behavioral_risk_score', 0)
                exposure = scores.get('sum_balance_avg', 0)
                
                segment_risk_ranking.append({
                    'segment': segment,
                    'risk_score': risk_score,
                    'exposure': exposure,
                    'risk_weighted_exposure': risk_score * exposure
                })
            
            segment_risk_ranking.sort(key=lambda x: x['risk_weighted_exposure'], reverse=True)
            behavioral_scores['segment_risk_ranking'] = segment_risk_ranking
        
        logger.info("Behavioral risk scores calculated")
        return behavioral_scores
        
    except Exception as e:
        logger.error(f"Error calculating behavioral risk scores: {e}")
        return {}

def calculate_liquidity_metrics(deposit_data: pd.DataFrame, market_data: pd.DataFrame = None) -> Dict:
    """Calculate comprehensive liquidity risk metrics"""
    try:
        logger.info("Calculating liquidity metrics...")
        
        liquidity_metrics = {}
        
        # 1. LCR calculation
        lcr_results = calculate_lcr_ratio(deposit_data)
        liquidity_metrics['lcr_analysis'] = lcr_results
        
        # 2. Deposit at Risk
        dar_results = calculate_deposit_at_risk(deposit_data)
        liquidity_metrics['deposit_at_risk'] = dar_results
        
        # 3. Deposit concentration metrics
        if 'balance' in deposit_data.columns:
            balances = deposit_data['balance']
            
            # Deposit distribution metrics
            deposit_metrics = {
                'total_deposits': balances.sum(),
                'avg_deposit': balances.mean(),
                'median_deposit': balances.median(),
                'deposit_concentration': {
                    'top_10_pct_share': balances.nlargest(int(len(balances) * 0.1)).sum() / balances.sum(),
                    'top_5_pct_share': balances.nlargest(int(len(balances) * 0.05)).sum() / balances.sum(),
                    'top_1_pct_share': balances.nlargest(int(len(balances) * 0.01)).sum() / balances.sum()
                },
                'gini_coefficient': calculate_gini_coefficient(balances)
            }
            
            liquidity_metrics['deposit_distribution'] = deposit_metrics
        
        # 4. Funding stability metrics
        if 'deposit_flow' in deposit_data.columns:
            flows = deposit_data['deposit_flow']
            
            # Flow stability metrics
            stability_metrics = {
                'flow_volatility': flows.std(),
                'flow_skewness': stats.skew(flows),
                'flow_kurtosis': stats.kurtosis(flows),
                'positive_flow_days': (flows > 0).sum() / len(flows),
                'large_outflow_days': (flows < -2 * flows.std()).sum() / len(flows),
                'max_single_day_outflow': flows.min(),
                'avg_outflow_on_stress_days': flows[flows < -flows.std()].mean()
            }
            
            liquidity_metrics['funding_stability'] = stability_metrics
        
        # 5. Market-based liquidity risk (if market data available)
        if market_data is not None and not market_data.empty:
            market_liquidity = calculate_market_liquidity_risk(market_data)
            liquidity_metrics['market_liquidity_risk'] = market_liquidity
        
        # 6. Early warning indicators
        warning_indicators = []
        
        if lcr_results.get('lcr_ratio', 1) < 1.1:
            warning_indicators.append("LCR below buffer threshold")
        
        if dar_results.get('backtesting_violations', 0) > 0.05:
            warning_indicators.append("High VaR violations")
        
        if deposit_metrics.get('deposit_concentration', {}).get('top_10_pct_share', 0) > 0.5:
            warning_indicators.append("High deposit concentration")
        
        liquidity_metrics['early_warnings'] = warning_indicators
        
        # 7. Overall liquidity risk score
        risk_score = calculate_overall_liquidity_risk_score(liquidity_metrics)
        liquidity_metrics['overall_risk_score'] = risk_score
        
        logger.info(f"Liquidity metrics calculated - Overall risk score: {risk_score:.2f}")
        return liquidity_metrics
        
    except Exception as e:
        logger.error(f"Error calculating liquidity metrics: {e}")
        return {}

def calculate_gini_coefficient(values: pd.Series) -> float:
    """Calculate Gini coefficient for concentration measurement"""
    try:
        sorted_values = sorted(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
        return gini
        
    except Exception as e:
        logger.error(f"Error calculating Gini coefficient: {e}")
        return 0.0

def calculate_market_liquidity_risk(market_data: pd.DataFrame) -> Dict:
    """Calculate market-based liquidity risk indicators"""
    try:
        market_liquidity = {}
        
        # VIX-based stress indicator
        if 'VIX' in market_data.columns:
            vix_current = market_data['VIX'].iloc[-1]
            vix_avg = market_data['VIX'].rolling(63).mean().iloc[-1]  # 3-month average
            
            market_liquidity['vix_analysis'] = {
                'current_vix': vix_current,
                'avg_vix_3m': vix_avg,
                'vix_stress_indicator': vix_current / vix_avg,
                'stress_level': 'High' if vix_current > 30 else 'Medium' if vix_current > 20 else 'Low'
            }
        
        # Interest rate environment
        if 'FED_FUNDS_RATE' in market_data.columns:
            fed_rate = market_data['FED_FUNDS_RATE'].iloc[-1]
            rate_change_30d = market_data['FED_FUNDS_RATE'].diff(30).iloc[-1]
            
            market_liquidity['rate_environment'] = {
                'current_rate': fed_rate,
                'rate_change_30d': rate_change_30d,
                'rate_regime': 'High' if fed_rate > 5 else 'Normal' if fed_rate > 1 else 'Low'
            }
        
        # Yield curve dynamics
        if all(col in market_data.columns for col in ['TREASURY_10Y', 'TREASURY_2Y']):
            yield_spread = market_data['TREASURY_10Y'].iloc[-1] - market_data['TREASURY_2Y'].iloc[-1]
            
            market_liquidity['yield_curve'] = {
                'current_spread': yield_spread,
                'curve_shape': 'Inverted' if yield_spread < 0 else 'Flat' if yield_spread < 0.5 else 'Normal'
            }
        
        return market_liquidity
        
    except Exception as e:
        logger.error(f"Error calculating market liquidity risk: {e}")
        return {}

def calculate_overall_liquidity_risk_score(liquidity_metrics: Dict) -> float:
    """Calculate overall liquidity risk score (0-10 scale, higher is riskier)"""
    try:
        risk_score = 0.0
        
        # LCR component (30% weight)
        lcr_ratio = liquidity_metrics.get('lcr_analysis', {}).get('lcr_ratio', 1.0)
        if lcr_ratio < 1.0:
            lcr_score = 4.0
        elif lcr_ratio < 1.1:
            lcr_score = 2.0
        else:
            lcr_score = 0.0
        risk_score += lcr_score * 0.3
        
        # Deposit concentration component (25% weight)
        top_10_share = liquidity_metrics.get('deposit_distribution', {}).get('deposit_concentration', {}).get('top_10_pct_share', 0.3)
        concentration_score = min(4.0, (top_10_share - 0.3) / 0.2 * 4.0) if top_10_share > 0.3 else 0.0
        risk_score += concentration_score * 0.25
        
        # Flow volatility component (20% weight)
        flow_volatility = liquidity_metrics.get('funding_stability', {}).get('flow_volatility', 0)
        if flow_volatility > 0:
            # Normalize volatility to 0-4 scale
            volatility_score = min(4.0, (flow_volatility / 1000000) * 2)  # Assuming $1M is high volatility
            risk_score += volatility_score * 0.2
        
        # VaR violations component (15% weight)
        var_violations = liquidity_metrics.get('deposit_at_risk', {}).get('backtesting_violations', 0)
        var_score = min(4.0, var_violations / 0.1 * 4.0) if var_violations > 0.05 else 0.0
        risk_score += var_score * 0.15
        
        # Early warnings component (10% weight)
        n_warnings = len(liquidity_metrics.get('early_warnings', []))
        warning_score = min(4.0, n_warnings * 1.5)
        risk_score += warning_score * 0.1
        
        return min(10.0, risk_score)
        
    except Exception as e:
        logger.error(f"Error calculating overall risk score: {e}")
        return 5.0  # Default medium risk

def generate_risk_report(deposit_data: pd.DataFrame, customer_data: pd.DataFrame, 
                        market_data: pd.DataFrame = None) -> Dict:
    """Generate comprehensive risk report"""
    try:
        logger.info("Generating comprehensive risk report...")
        
        risk_report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': {
                'start_date': deposit_data['date'].min() if 'date' in deposit_data.columns else 'N/A',
                'end_date': deposit_data['date'].max() if 'date' in deposit_data.columns else 'N/A',
                'n_observations': len(deposit_data)
            }
        }
        
        # Core liquidity metrics
        risk_report['liquidity_metrics'] = calculate_liquidity_metrics(deposit_data, market_data)
        
        # Concentration analysis
        risk_report['concentration_risk'] = calculate_concentration_risk(customer_data, deposit_data)
        
        # Behavioral risk analysis
        risk_report['behavioral_risk'] = calculate_behavioral_risk_scores(customer_data, deposit_data)
        
        # Risk summary and recommendations
        risk_report['executive_summary'] = generate_executive_summary(risk_report)
        
        logger.info("Risk report generated successfully")
        return risk_report
        
    except Exception as e:
        logger.error(f"Error generating risk report: {e}")
        return {}

def generate_executive_summary(risk_report: Dict) -> Dict:
    """Generate executive summary with key findings and recommendations"""
    try:
        summary = {
            'key_metrics': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Extract key metrics
        liquidity_metrics = risk_report.get('liquidity_metrics', {})
        lcr_ratio = liquidity_metrics.get('lcr_analysis', {}).get('lcr_ratio', 1.0)
        overall_risk_score = liquidity_metrics.get('overall_risk_score', 5.0)
        
        summary['key_metrics'] = {
            'lcr_ratio': f"{lcr_ratio:.1%}",
            'overall_risk_score': f"{overall_risk_score:.1f}/10",
            'risk_level': 'High' if overall_risk_score > 7 else 'Medium' if overall_risk_score > 4 else 'Low'
        }
        
        # Risk assessment
        early_warnings = liquidity_metrics.get('early_warnings', [])
        summary['risk_assessment'] = {
            'immediate_concerns': early_warnings,
            'n_risk_flags': len(early_warnings)
        }
        
        # Generate recommendations
        recommendations = []
        
        if lcr_ratio < 1.1:
            recommendations.append("Increase liquid asset buffer to improve LCR ratio")
        
        if len(early_warnings) > 2:
            recommendations.append("Implement enhanced monitoring due to multiple risk flags")
        
        concentration_risk = risk_report.get('concentration_risk', {})
        if concentration_risk.get('risk_assessment', {}).get('overall_risk') == 'High':
            recommendations.append("Diversify deposit base to reduce concentration risk")
        
        if overall_risk_score > 6:
            recommendations.append("Consider stress testing and contingency funding plan review")
        
        if not recommendations:
            recommendations.append("Current risk profile is acceptable, maintain monitoring")
        
        summary['recommendations'] = recommendations
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("Risk calculator module loaded successfully")
    print("Available functions:")
    print("- calculate_lcr_ratio")
    print("- calculate_deposit_at_risk")
    print("- calculate_concentration_risk")
    print("- calculate_behavioral_risk_scores")
    print("- calculate_liquidity_metrics")
    print("- generate_risk_report")