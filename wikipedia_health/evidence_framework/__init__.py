"""Evidence Framework.

Orchestrates cross-validation and robustness checks.
"""

from wikipedia_health.evidence_framework.cross_validator import (
    CrossValidator,
    ValidationResult,
    ComparisonResult
)
from wikipedia_health.evidence_framework.robustness_checker import (
    RobustnessChecker,
    SensitivityResult,
    OutlierSensitivityResult,
    MethodComparisonResult,
    StabilityResult
)

__all__ = [
    'CrossValidator',
    'ValidationResult',
    'ComparisonResult',
    'RobustnessChecker',
    'SensitivityResult',
    'OutlierSensitivityResult',
    'MethodComparisonResult',
    'StabilityResult'
]
