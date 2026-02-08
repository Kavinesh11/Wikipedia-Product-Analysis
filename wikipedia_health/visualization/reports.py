"""Report generation functions for analysis results.

This module provides functions for generating summary tables, finding reports,
and evidence reports with plain-language interpretations.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from pandas import DataFrame
from datetime import datetime

from wikipedia_health.models import (
    TestResult,
    CausalEffect,
    Finding,
    ForecastResult,
    DecompositionResult,
)


def generate_summary_table(
    test_results: List[TestResult],
    causal_effects: Optional[List[CausalEffect]] = None,
    format: str = 'dataframe'
) -> DataFrame:
    """Generate summary table with test statistics, p-values, CIs, and effect sizes.
    
    Args:
        test_results: List of TestResult objects to summarize
        causal_effects: Optional list of CausalEffect objects to include
        format: Output format ('dataframe', 'html', 'markdown')
        
    Returns:
        DataFrame with summary statistics
        
    Requirements: 13.6
    """
    # Build rows for statistical tests
    rows = []
    
    for test in test_results:
        # Determine significance indicator
        if test.p_value < 0.001:
            sig_indicator = "***"
        elif test.p_value < 0.01:
            sig_indicator = "**"
        elif test.p_value < 0.05:
            sig_indicator = "*"
        else:
            sig_indicator = ""
        
        # Interpret effect size (Cohen's d conventions)
        if abs(test.effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(test.effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(test.effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        row = {
            'Test': test.test_name,
            'Statistic': f"{test.statistic:.4f}",
            'p-value': f"{test.p_value:.4f}{sig_indicator}",
            'Significant': 'Yes' if test.is_significant else 'No',
            'Effect Size': f"{test.effect_size:.4f}",
            'Effect Magnitude': effect_interpretation,
            '95% CI Lower': f"{test.confidence_interval[0]:.4f}",
            '95% CI Upper': f"{test.confidence_interval[1]:.4f}",
            'Interpretation': test.interpretation
        }
        rows.append(row)
    
    # Add causal effects if provided
    if causal_effects:
        for effect in causal_effects:
            # Determine significance indicator
            if effect.p_value < 0.001:
                sig_indicator = "***"
            elif effect.p_value < 0.01:
                sig_indicator = "**"
            elif effect.p_value < 0.05:
                sig_indicator = "*"
            else:
                sig_indicator = ""
            
            row = {
                'Test': f"Causal Effect ({effect.method})",
                'Statistic': f"{effect.effect_size:.4f}",
                'p-value': f"{effect.p_value:.4f}{sig_indicator}",
                'Significant': 'Yes' if effect.p_value < 0.05 else 'No',
                'Effect Size': f"{effect.percentage_effect():.2f}%",
                'Effect Magnitude': 'causal',
                '95% CI Lower': f"{effect.confidence_interval[0]:.4f}",
                '95% CI Upper': f"{effect.confidence_interval[1]:.4f}",
                'Interpretation': f"Treatment period: {effect.treatment_period[0]} to {effect.treatment_period[1]}"
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Format based on requested output
    if format == 'dataframe':
        return df
    elif format == 'html':
        return df.to_html(index=False, escape=False)
    elif format == 'markdown':
        return df.to_markdown(index=False)
    else:
        return df


def generate_finding_report(
    finding: Finding,
    include_evidence_details: bool = True,
    include_recommendations: bool = True
) -> str:
    """Generate plain-language finding report with interpretations.
    
    Args:
        finding: Finding object to report on
        include_evidence_details: Whether to include detailed evidence breakdown
        include_recommendations: Whether to include actionable recommendations
        
    Returns:
        Formatted string report
        
    Requirements: 13.6
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(f"FINDING REPORT: {finding.finding_id}")
    lines.append("=" * 80)
    lines.append("")
    
    # Confidence level with visual indicator
    confidence_indicators = {
        'high': '🟢 HIGH',
        'medium': '🟡 MEDIUM',
        'low': '🔴 LOW'
    }
    confidence_display = confidence_indicators.get(
        finding.confidence_level.lower(),
        finding.confidence_level.upper()
    )
    lines.append(f"Confidence Level: {confidence_display}")
    lines.append(f"Evidence Strength Score: {finding.evidence_strength():.2f}/1.00")
    lines.append("")
    
    # Description
    lines.append("DESCRIPTION")
    lines.append("-" * 80)
    lines.append(finding.description)
    lines.append("")
    
    # Statistical Evidence
    if finding.evidence:
        lines.append("STATISTICAL EVIDENCE")
        lines.append("-" * 80)
        lines.append(f"Total Tests Conducted: {len(finding.evidence)}")
        
        significant_tests = [t for t in finding.evidence if t.is_significant]
        lines.append(f"Significant Results: {len(significant_tests)}/{len(finding.evidence)}")
        lines.append("")
        
        if include_evidence_details:
            for i, test in enumerate(finding.evidence, 1):
                # Significance indicator
                if test.p_value < 0.001:
                    sig_marker = "*** (p < 0.001)"
                elif test.p_value < 0.01:
                    sig_marker = "** (p < 0.01)"
                elif test.p_value < 0.05:
                    sig_marker = "* (p < 0.05)"
                else:
                    sig_marker = "(not significant)"
                
                lines.append(f"{i}. {test.test_name} {sig_marker}")
                lines.append(f"   Test Statistic: {test.statistic:.4f}")
                lines.append(f"   P-value: {test.p_value:.6f}")
                lines.append(f"   Effect Size: {test.effect_size:.4f}")
                lines.append(f"   95% CI: [{test.confidence_interval[0]:.4f}, {test.confidence_interval[1]:.4f}]")
                lines.append(f"   Interpretation: {test.interpretation}")
                lines.append("")
    
    # Causal Effects
    if finding.causal_effects:
        lines.append("CAUSAL ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Causal Methods Applied: {len(finding.causal_effects)}")
        lines.append("")
        
        for i, effect in enumerate(finding.causal_effects, 1):
            # Significance indicator
            if effect.p_value < 0.001:
                sig_marker = "*** (p < 0.001)"
            elif effect.p_value < 0.01:
                sig_marker = "** (p < 0.01)"
            elif effect.p_value < 0.05:
                sig_marker = "* (p < 0.05)"
            else:
                sig_marker = "(not significant)"
            
            lines.append(f"{i}. {effect.method} {sig_marker}")
            lines.append(f"   Causal Effect Size: {effect.effect_size:.4f}")
            lines.append(f"   Percentage Effect: {effect.percentage_effect():+.2f}%")
            lines.append(f"   P-value: {effect.p_value:.6f}")
            lines.append(f"   95% CI: [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}]")
            lines.append(f"   Treatment Period: {effect.treatment_period[0]} to {effect.treatment_period[1]}")
            lines.append("")
    
    # Requirements Validated
    if finding.requirements_validated:
        lines.append("REQUIREMENTS VALIDATED")
        lines.append("-" * 80)
        lines.append(", ".join(finding.requirements_validated))
        lines.append("")
    
    # Recommendations
    if include_recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        
        # Generate recommendations based on confidence and evidence
        if finding.confidence_level.lower() == 'high':
            lines.append("✓ This finding is well-supported by statistical evidence.")
            lines.append("✓ Recommended for decision-making and strategic planning.")
        elif finding.confidence_level.lower() == 'medium':
            lines.append("⚠ This finding has moderate statistical support.")
            lines.append("⚠ Consider additional validation before major decisions.")
        else:
            lines.append("⚠ This finding has limited statistical support.")
            lines.append("⚠ Recommend further investigation and data collection.")
        
        lines.append("")
        
        # Specific recommendations based on evidence
        if finding.evidence:
            non_sig = [t for t in finding.evidence if not t.is_significant]
            if non_sig:
                lines.append(f"• {len(non_sig)} test(s) did not reach significance - consider alternative hypotheses.")
        
        if finding.causal_effects:
            for effect in finding.causal_effects:
                if effect.p_value >= 0.05:
                    lines.append(f"• {effect.method} did not establish causal significance - interpret with caution.")
        
        lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_evidence_report(
    findings: List[Finding],
    title: str = "Evidence Summary Report",
    include_cross_validation: bool = True
) -> str:
    """Generate comprehensive evidence report aggregating all statistical evidence.
    
    Args:
        findings: List of Finding objects to aggregate
        title: Report title
        include_cross_validation: Whether to include cross-validation summary
        
    Returns:
        Formatted string report with aggregated evidence
        
    Requirements: 13.6
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(title.upper())
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total Findings: {len(findings)}")
    
    # Count by confidence level
    confidence_counts = {}
    for finding in findings:
        level = finding.confidence_level.lower()
        confidence_counts[level] = confidence_counts.get(level, 0) + 1
    
    lines.append(f"High Confidence: {confidence_counts.get('high', 0)}")
    lines.append(f"Medium Confidence: {confidence_counts.get('medium', 0)}")
    lines.append(f"Low Confidence: {confidence_counts.get('low', 0)}")
    lines.append("")
    
    # Aggregate statistics
    total_tests = sum(len(f.evidence) for f in findings)
    total_significant = sum(
        len([t for t in f.evidence if t.is_significant])
        for f in findings
    )
    total_causal = sum(len(f.causal_effects) for f in findings)
    
    lines.append(f"Total Statistical Tests: {total_tests}")
    if total_tests > 0:
        lines.append(f"Significant Results: {total_significant}/{total_tests} ({total_significant/total_tests*100:.1f}%)")
    else:
        lines.append(f"Significant Results: 0/0 (N/A)")
    lines.append(f"Causal Analyses: {total_causal}")
    lines.append("")
    
    # Average evidence strength
    avg_strength = sum(f.evidence_strength() for f in findings) / len(findings) if findings else 0
    lines.append(f"Average Evidence Strength: {avg_strength:.2f}/1.00")
    lines.append("")
    
    # Individual Findings Summary
    lines.append("FINDINGS SUMMARY")
    lines.append("-" * 80)
    
    for i, finding in enumerate(findings, 1):
        confidence_emoji = {
            'high': '🟢',
            'medium': '🟡',
            'low': '🔴'
        }.get(finding.confidence_level.lower(), '⚪')
        
        lines.append(f"{i}. [{finding.finding_id}] {confidence_emoji} {finding.confidence_level.upper()}")
        lines.append(f"   {finding.description[:100]}...")
        lines.append(f"   Evidence: {len(finding.evidence)} tests, {len(finding.causal_effects)} causal analyses")
        lines.append(f"   Strength: {finding.evidence_strength():.2f}/1.00")
        
        # Show key statistics
        if finding.evidence:
            sig_tests = [t for t in finding.evidence if t.is_significant]
            lines.append(f"   Significant Tests: {len(sig_tests)}/{len(finding.evidence)}")
        
        lines.append("")
    
    # Cross-Validation Summary
    if include_cross_validation:
        lines.append("CROSS-VALIDATION SUMMARY")
        lines.append("-" * 80)
        
        # Collect all requirements validated
        all_requirements = set()
        for finding in findings:
            all_requirements.update(finding.requirements_validated)
        
        lines.append(f"Requirements Validated: {len(all_requirements)}")
        if all_requirements:
            lines.append(f"Requirement IDs: {', '.join(sorted(all_requirements))}")
        lines.append("")
        
        # Method diversity
        all_methods = set()
        for finding in findings:
            for effect in finding.causal_effects:
                all_methods.add(effect.method)
        
        if all_methods:
            lines.append(f"Causal Methods Used: {', '.join(sorted(all_methods))}")
            lines.append("")
    
    # Recommendations
    lines.append("OVERALL RECOMMENDATIONS")
    lines.append("-" * 80)
    
    high_conf = confidence_counts.get('high', 0)
    low_conf = confidence_counts.get('low', 0)
    
    if high_conf >= len(findings) * 0.7:
        lines.append("✓ Strong evidence base - findings are well-supported for decision-making.")
    elif low_conf >= len(findings) * 0.5:
        lines.append("⚠ Weak evidence base - recommend additional data collection and analysis.")
    else:
        lines.append("⚠ Mixed evidence base - prioritize high-confidence findings for decisions.")
    
    lines.append("")
    
    if total_significant / total_tests < 0.5 if total_tests > 0 else False:
        lines.append("⚠ Low proportion of significant results - consider alternative hypotheses.")
    
    if total_causal == 0:
        lines.append("⚠ No causal analyses performed - correlational findings only.")
    
    lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def export_report_to_file(
    report_content: str,
    filename: str,
    format: str = 'txt'
) -> None:
    """Export report to file.
    
    Args:
        report_content: Report content as string
        filename: Output filename (without extension)
        format: Output format ('txt', 'md', 'html')
    """
    full_filename = f"{filename}.{format}"
    
    with open(full_filename, 'w', encoding='utf-8') as f:
        if format == 'html':
            # Wrap in basic HTML
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Report</title>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 40px; }}
        pre {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
    <pre>{report_content}</pre>
</body>
</html>
"""
            f.write(html_content)
        else:
            f.write(report_content)
