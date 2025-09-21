import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from config import RISK_CONFIG, CUSTOMER_CONFIG
from src.analytics.risk_calculator import calculate_lcr_ratio, calculate_deposit_at_risk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_scenario_analysis(base_data: Dict, scenarios: Dict) -> Dict:
    """Run comprehensive scenario analysis on deposit and liquidity data"""
    try:
        logger.info("Running scenario analysis...")
        
        scenario_results = {
            'base_case': calculate_base_case_metrics(base_data),
            'scenario_results': {},
            'comparative_analysis': {},
            'risk_assessment': {}
        }
        
        # Run each scenario
        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Running scenario: {scenario_name}")
            
            scenario_data = apply_scenario_shocks(base_data, scenario_params)
            scenario_metrics = calculate_scenario_metrics(scenario_data)
            
            scenario_results['scenario_results'][scenario_name] = {
                'parameters': scenario_params,
                'metrics': scenario_metrics,
                'data': scenario_data
            }
        
        # Comparative analysis
        scenario_results['comparative_analysis'] = compare_scenarios(
            scenario_results['base_case'],
            scenario_results['scenario_results']
        )
        
        # Risk assessment
        scenario_results['risk_assessment'] = assess_scenario_risks(
            scenario_results['scenario_results']
        )
        
        logger.info("Scenario analysis completed")
        return scenario_results
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        return {}

def calculate_base_case_metrics(base_data: Dict) -> Dict:
    """Calculate baseline metrics for comparison"""
    try:
        base_metrics = {}
        
        # Extract data
        deposit_data = base_data.get('deposit_data', pd.DataFrame())
        customer_data = base_data.get('customer_data', pd.DataFrame())
        market_data = base_data.get('market_data', pd.DataFrame())
        
        if not deposit_data.empty:
            # Basic deposit metrics
            base_metrics['total_deposits'] = deposit_data['balance'].sum() if 'balance' in deposit_data.columns else 0
            base_metrics['avg_daily_flow'] = deposit_data['deposit_flow'].mean() if 'deposit_flow' in deposit_data.columns else 0
            base_metrics['flow_volatility'] = deposit_data['deposit_flow'].std() if 'deposit_flow' in deposit_data.columns else 0
            
            # LCR calculation
            lcr_results = calculate_lcr_ratio(deposit_data)
            base_metrics['lcr_ratio'] = lcr_results.get('lcr_ratio', 1.0)
            
            # Deposit at Risk
            dar_results = calculate_deposit_at_risk(deposit_data)
            base_metrics['deposit_at_risk_95'] = dar_results.get('historical_var', 0)
        
        # Customer segment metrics
        if not customer_data.empty:
            segment_metrics = customer_data.groupby('segment').agg({
                'balance_avg': ['sum', 'count', 'mean'],
                'rate_sensitivity': 'mean',
                'loyalty_score': 'mean'
            }).round(4)
            
            base_metrics['segment_breakdown'] = segment_metrics.to_dict()
        
        # Market conditions
        if not market_data.empty:
            base_metrics['market_conditions'] = {
                'fed_funds_rate': market_data['FED_FUNDS_RATE'].iloc[-1] if 'FED_FUNDS_RATE' in market_data.columns else 2.0,
                'treasury_10y': market_data['TREASURY_10Y'].iloc[-1] if 'TREASURY_10Y' in market_data.columns else 3.0,
                'vix_level': market_data['VIX'].iloc[-1] if 'VIX' in market_data.columns else 20.0
            }
        
        return base_metrics
        
    except Exception as e:
        logger.error(f"Error calculating base case metrics: {e}")
        return {}

def apply_scenario_shocks(base_data: Dict, scenario_params: Dict) -> Dict:
    """Apply scenario shocks to base data"""
    try:
        scenario_data = {}
        
        # Copy base data
        deposit_data = base_data.get('deposit_data', pd.DataFrame()).copy()
        customer_data = base_data.get('customer_data', pd.DataFrame()).copy()
        market_data = base_data.get('market_data', pd.DataFrame()).copy()
        
        # Apply interest rate shock
        rate_shock = scenario_params.get('interest_rate_shock', 0)  # in basis points
        if rate_shock != 0 and not market_data.empty:
            if 'FED_FUNDS_RATE' in market_data.columns:
                market_data['FED_FUNDS_RATE'] += rate_shock / 100
            if 'TREASURY_10Y' in market_data.columns:
                market_data['TREASURY_10Y'] += rate_shock / 100 * 0.8  # Correlated but dampened
        
        # Apply market stress (VIX shock)
        vix_shock = scenario_params.get('vix_shock', 0)
        if vix_shock != 0 and not market_data.empty and 'VIX' in market_data.columns:
            market_data['VIX'] = np.maximum(10, market_data['VIX'] + vix_shock)
        
        # Apply deposit outflow shock
        deposit_outflow_rate = scenario_params.get('deposit_outflow_rate', 0)
        if deposit_outflow_rate > 0 and not deposit_data.empty:
            # Apply differentiated outflows by customer segment
            segment_outflows = apply_behavioral_outflows(
                customer_data, deposit_data, deposit_outflow_rate, rate_shock
            )
            deposit_data = segment_outflows
        
        # Apply credit spread shock
        credit_spread_shock = scenario_params.get('credit_spread_shock', 0)
        if credit_spread_shock != 0:
            # This would affect funding costs and liquidity requirements
            pass  # Implementation depends on specific credit data availability
        
        # Economic shock impacts
        economic_shock = scenario_params.get('economic_shock', 'none')
        if economic_shock != 'none':
            deposit_data, customer_data = apply_economic_shock(
                deposit_data, customer_data, economic_shock
            )
        
        scenario_data = {
            'deposit_data': deposit_data,
            'customer_data': customer_data,
            'market_data': market_data,
            'shocks_applied': scenario_params
        }
        
        return scenario_data
        
    except Exception as e:
        logger.error(f"Error applying scenario shocks: {e}")
        return base_data

def apply_behavioral_outflows(customer_data: pd.DataFrame, deposit_data: pd.DataFrame, 
                             base_outflow_rate: float, rate_shock: float) -> pd.DataFrame:
    """Apply behavioral deposit outflows based on customer characteristics"""
    try:
        shocked_data = deposit_data.copy()
        
        if customer_data.empty:
            # Simple uniform outflow if no customer data
            if 'balance' in shocked_data.columns:
                shocked_data['balance'] *= (1 - base_outflow_rate)
            return shocked_data
        
        # Calculate segment-specific outflow rates
        segment_outflows = {}
        
        for segment in customer_data['segment'].unique():
            segment_customers = customer_data[customer_data['segment'] == segment]
            
            # Base outflow adjusted by behavioral factors
            avg_rate_sensitivity = segment_customers['rate_sensitivity'].mean()
            avg_loyalty = segment_customers['loyalty_score'].mean()
            avg_volatility = segment_customers['volatility_factor'].mean()
            
            # Outflow multiplier based on behavioral characteristics
            rate_impact = 1 + (avg_rate_sensitivity * abs(rate_shock) / 100)
            loyalty_buffer = avg_loyalty  # Higher loyalty reduces outflows
            volatility_multiplier = 1 + avg_volatility
            
            segment_outflow_rate = base_outflow_rate * rate_impact * volatility_multiplier * (2 - loyalty_buffer)
            segment_outflow_rate = min(0.8, segment_outflow_rate)  # Cap at 80% outflow
            
            segment_outflows[segment] = segment_outflow_rate
        
        # Apply outflows to deposit data
        if 'segment' in shocked_data.columns and 'balance' in shocked_data.columns:
            for segment, outflow_rate in segment_outflows.items():
                mask = shocked_data['segment'] == segment
                shocked_data.loc[mask, 'balance'] *= (1 - outflow_rate)
                
                # Update deposit flows
                if 'deposit_flow' in shocked_data.columns:
                    outflow_amount = shocked_data.loc[mask, 'balance'] * outflow_rate
                    shocked_data.loc[mask, 'deposit_flow'] = -outflow_amount
        
        return shocked_data
        
    except Exception as e:
        logger.error(f"Error applying behavioral outflows: {e}")
        return deposit_data

def apply_economic_shock(deposit_data: pd.DataFrame, customer_data: pd.DataFrame, 
                        shock_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply macroeconomic shock scenarios"""
    try:
        shocked_deposit_data = deposit_data.copy()
        shocked_customer_data = customer_data.copy()
        
        shock_parameters = {
            'mild_recession': {
                'deposit_impact': 0.95,  # 5% deposit reduction
                'flow_volatility_multiplier': 1.3,
                'rate_sensitivity_increase': 0.1
            },
            'severe_recession': {
                'deposit_impact': 0.85,  # 15% deposit reduction
                'flow_volatility_multiplier': 2.0,
                'rate_sensitivity_increase': 0.2
            },
            'financial_crisis': {
                'deposit_impact': 0.75,  # 25% deposit reduction
                'flow_volatility_multiplier': 3.0,
                'rate_sensitivity_increase': 0.3
            },
            'inflation_spike': {
                'deposit_impact': 0.90,  # 10% deposit reduction
                'flow_volatility_multiplier': 1.5,
                'rate_sensitivity_increase': 0.25
            }
        }
        
        if shock_type not in shock_parameters:
            logger.warning(f"Unknown shock type: {shock_type}")
            return deposit_data, customer_data
        
        params = shock_parameters[shock_type]
        
        # Apply deposit impact
        if 'balance' in shocked_deposit_data.columns:
            shocked_deposit_data['balance'] *= params['deposit_impact']
        
        # Increase flow volatility
        if 'deposit_flow' in shocked_deposit_data.columns:
            flow_std = shocked_deposit_data['deposit_flow'].std()
            noise = np.random.normal(0, flow_std * (params['flow_volatility_multiplier'] - 1), 
                                   len(shocked_deposit_data))
            shocked_deposit_data['deposit_flow'] += noise
        
        # Update customer behavioral parameters
        if 'rate_sensitivity' in shocked_customer_data.columns:
            shocked_customer_data['rate_sensitivity'] = np.minimum(
                1.0, 
                shocked_customer_data['rate_sensitivity'] + params['rate_sensitivity_increase']
            )
        
        return shocked_deposit_data, shocked_customer_data
        
    except Exception as e:
        logger.error(f"Error applying economic shock: {e}")
        return deposit_data, customer_data

def calculate_scenario_metrics(scenario_data: Dict) -> Dict:
    """Calculate key metrics for scenario results"""
    try:
        metrics = {}
        
        deposit_data = scenario_data.get('deposit_data', pd.DataFrame())
        customer_data = scenario_data.get('customer_data', pd.DataFrame())
        market_data = scenario_data.get('market_data', pd.DataFrame())
        
        if not deposit_data.empty:
            # Deposit metrics
            metrics['total_deposits'] = deposit_data['balance'].sum() if 'balance' in deposit_data.columns else 0
            metrics['deposit_change_pct'] = 0  # Will be calculated in comparison
            
            # Flow metrics
            if 'deposit_flow' in deposit_data.columns:
                metrics['avg_daily_flow'] = deposit_data['deposit_flow'].mean()
                metrics['flow_volatility'] = deposit_data['deposit_flow'].std()
                metrics['max_daily_outflow'] = deposit_data['deposit_flow'].min()
                metrics['outflow_days_count'] = (deposit_data['deposit_flow'] < 0).sum()
            
            # LCR under stress
            lcr_results = calculate_lcr_ratio(deposit_data)
            metrics['stressed_lcr_ratio'] = lcr_results.get('lcr_ratio', 0)
            metrics['lcr_breach'] = lcr_results.get('lcr_ratio', 1) < 1.0
            
            # Deposit at Risk under stress
            dar_results = calculate_deposit_at_risk(deposit_data)
            metrics['stressed_dar_95'] = dar_results.get('historical_var', 0)
        
        # Funding gap analysis
        if metrics.get('stressed_lcr_ratio', 1) < 1.0:
            required_liquidity = metrics.get('max_daily_outflow', 0) * 30  # 30-day coverage
            available_liquidity = metrics.get('total_deposits', 0) * 0.15  # Assume 15% liquid assets
            metrics['funding_gap'] = max(0, required_liquidity - available_liquidity)
        else:
            metrics['funding_gap'] = 0
        
        # Segment-level stress impacts
        if not customer_data.empty and 'segment' in customer_data.columns:
            segment_impacts = customer_data.groupby('segment').agg({
                'balance_avg': 'sum'
            })
            metrics['segment_impacts'] = segment_impacts.to_dict()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating scenario metrics: {e}")
        return {}

def compare_scenarios(base_case: Dict, scenario_results: Dict) -> Dict:
    """Compare scenario results against base case"""
    try:
        comparisons = {}
        
        base_deposits = base_case.get('total_deposits', 1)
        base_lcr = base_case.get('lcr_ratio', 1)
        base_dar = base_case.get('deposit_at_risk_95', 0)
        
        for scenario_name, scenario_data in scenario_results.items():
            metrics = scenario_data.get('metrics', {})
            
            comparison = {
                'deposit_change_abs': metrics.get('total_deposits', 0) - base_deposits,
                'deposit_change_pct': (metrics.get('total_deposits', 0) - base_deposits) / base_deposits * 100,
                'lcr_change': metrics.get('stressed_lcr_ratio', 0) - base_lcr,
                'lcr_breach_risk': metrics.get('lcr_breach', False),
                'dar_change': metrics.get('stressed_dar_95', 0) - base_dar,
                'funding_gap': metrics.get('funding_gap', 0)
            }
            
            # Risk assessment
            if comparison['deposit_change_pct'] < -20:
                comparison['risk_level'] = 'High'
            elif comparison['deposit_change_pct'] < -10:
                comparison['risk_level'] = 'Medium'
            else:
                comparison['risk_level'] = 'Low'
            
            comparisons[scenario_name] = comparison
        
        logger.info(f"Compared {len(scenario_results)} scenarios against base case")
        return comparisons
        
    except Exception as e:
        logger.error(f"Error comparing scenarios: {e}")
        return {}

def assess_scenario_risks(scenario_results: Dict) -> Dict:
    """Assess overall risk implications of scenarios"""
    try:
        risk_assessment = {
            'highest_risk_scenario': None,
            'lowest_risk_scenario': None,
            'critical_scenarios': [],
            'risk_factors': {}
        }
        
        scenario_risks = {}
        
        for scenario_name, scenario_data in scenario_results.items():
            metrics = scenario_data.get('metrics', {})
            
            # Calculate composite risk score
            risk_score = 0
            
            # LCR breach (40% weight)
            if metrics.get('lcr_breach', False):
                risk_score += 40
            elif metrics.get('stressed_lcr_ratio', 1) < 1.1:
                risk_score += 20
            
            # Deposit loss (30% weight)
            deposit_loss = abs(metrics.get('deposit_change_pct', 0))
            if deposit_loss > 30:
                risk_score += 30
            elif deposit_loss > 15:
                risk_score += 20
            elif deposit_loss > 5:
                risk_score += 10
            
            # Funding gap (20% weight)
            funding_gap = metrics.get('funding_gap', 0)
            if funding_gap > 1e9:  # $1B
                risk_score += 20
            elif funding_gap > 5e8:  # $500M
                risk_score += 15
            elif funding_gap > 1e8:  # $100M
                risk_score += 10
            
            # Flow volatility (10% weight)
            flow_vol = metrics.get('flow_volatility', 0)
            base_vol = 1e6  # Assume $1M base volatility
            if flow_vol > base_vol * 3:
                risk_score += 10
            elif flow_vol > base_vol * 2:
                risk_score += 5
            
            scenario_risks[scenario_name] = risk_score
            
            # Flag critical scenarios
            if risk_score >= 70:
                risk_assessment['critical_scenarios'].append({
                    'name': scenario_name,
                    'risk_score': risk_score,
                    'issues': []
                })
        
        # Find highest and lowest risk scenarios
        if scenario_risks:
            risk_assessment['highest_risk_scenario'] = max(scenario_risks, key=scenario_risks.get)
            risk_assessment['lowest_risk_scenario'] = min(scenario_risks, key=scenario_risks.get)
        
        # Risk factors analysis
        risk_assessment['risk_factors'] = {
            'lcr_breaches': sum(1 for s in scenario_results.values() 
                               if s.get('metrics', {}).get('lcr_breach', False)),
            'high_deposit_loss': sum(1 for s in scenario_results.values() 
                                   if abs(s.get('metrics', {}).get('deposit_change_pct', 0)) > 20),
            'funding_gaps': sum(1 for s in scenario_results.values() 
                              if s.get('metrics', {}).get('funding_gap', 0) > 1e8)
        }
        
        logger.info("Risk assessment completed")
        return risk_assessment
        
    except Exception as e:
        logger.error(f"Error assessing scenario risks: {e}")
        return {}

def generate_scenario_report(scenario_results: Dict) -> Dict:
    """Generate comprehensive scenario analysis report"""
    try:
        report = {
            'executive_summary': {},
            'scenario_results': scenario_results,
            'key_findings': [],
            'recommendations': [],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Executive summary
        base_case = scenario_results.get('base_case', {})
        comparisons = scenario_results.get('comparative_analysis', {})
        risk_assessment = scenario_results.get('risk_assessment', {})
        
        report['executive_summary'] = {
            'scenarios_analyzed': len(scenario_results.get('scenario_results', {})),
            'critical_scenarios': len(risk_assessment.get('critical_scenarios', [])),
            'worst_case_scenario': risk_assessment.get('highest_risk_scenario', 'Unknown'),
            'base_case_lcr': base_case.get('lcr_ratio', 1.0)
        }
        
        # Key findings
        key_findings = []
        
        # LCR findings
        lcr_breaches = risk_assessment.get('risk_factors', {}).get('lcr_breaches', 0)
        if lcr_breaches > 0:
            key_findings.append(f"LCR breaches occur in {lcr_breaches} scenario(s)")
        
        # Deposit loss findings
        high_loss_scenarios = risk_assessment.get('risk_factors', {}).get('high_deposit_loss', 0)
        if high_loss_scenarios > 0:
            key_findings.append(f"Significant deposit losses (>20%) in {high_loss_scenarios} scenario(s)")
        
        # Funding gap findings
        funding_gap_scenarios = risk_assessment.get('risk_factors', {}).get('funding_gaps', 0)
        if funding_gap_scenarios > 0:
            key_findings.append(f"Funding gaps identified in {funding_gap_scenarios} scenario(s)")
        
        report['key_findings'] = key_findings
        
        # Recommendations
        recommendations = []
        
        if lcr_breaches > 0:
            recommendations.append("Increase liquid asset buffer to maintain LCR above regulatory minimum")
        
        if high_loss_scenarios > 0:
            recommendations.append("Develop deposit retention strategies for high-risk scenarios")
        
        if funding_gap_scenarios > 0:
            recommendations.append("Establish contingent funding sources for stress scenarios")
        
        worst_scenario = risk_assessment.get('highest_risk_scenario')
        if worst_scenario:
            recommendations.append(f"Develop specific action plan for {worst_scenario} scenario")
        
        if not recommendations:
            recommendations.append("Current liquidity position appears resilient to analyzed scenarios")
        
        report['recommendations'] = recommendations
        
        logger.info("Scenario analysis report generated")
        return report
        
    except Exception as e:
        logger.error(f"Error generating scenario report: {e}")
        return {}

# Predefined scenario templates
def get_standard_scenarios() -> Dict:
    """Get standard scenario templates for testing"""
    return {
        'mild_rate_shock': {
            'interest_rate_shock': 100,  # 100 bps
            'deposit_outflow_rate': 0.05,
            'vix_shock': 10,
            'economic_shock': 'none'
        },
        'severe_rate_shock': {
            'interest_rate_shock': 300,  # 300 bps
            'deposit_outflow_rate': 0.20,
            'vix_shock': 20,
            'economic_shock': 'mild_recession'
        },
        'financial_crisis': {
            'interest_rate_shock': 0,
            'deposit_outflow_rate': 0.30,
            'vix_shock': 40,
            'economic_shock': 'financial_crisis'
        },
        'economic_recession': {
            'interest_rate_shock': -200,  # Rate cuts
            'deposit_outflow_rate': 0.15,
            'vix_shock': 25,
            'economic_shock': 'severe_recession'
        }
    }

if __name__ == "__main__":
    # Example usage
    logger.info("Scenario analyzer module loaded successfully")
    print("Available functions:")
    print("- run_scenario_analysis")
    print("- calculate_base_case_metrics")
    print("- apply_scenario_shocks")
    print("- calculate_scenario_metrics")
    print("- compare_scenarios")
    print("- assess_scenario_risks")
    print("- generate_scenario_report")
    print("- get_standard_scenarios")