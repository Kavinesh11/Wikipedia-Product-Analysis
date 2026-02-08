"""Statistical validation module for Wikipedia product health analysis.

This module provides statistical hypothesis testing, confidence interval
calculation, and effect size computation for validating findings.
"""

from wikipedia_health.statistical_validation.hypothesis_tester import HypothesisTester
from wikipedia_health.statistical_validation.confidence_interval import ConfidenceIntervalCalculator
from wikipedia_health.statistical_validation.effect_size import EffectSizeCalculator

__all__ = [
    'HypothesisTester',
    'ConfidenceIntervalCalculator',
    'EffectSizeCalculator',
]
