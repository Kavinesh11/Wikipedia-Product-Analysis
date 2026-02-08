"""Analysis Logger for Reproducibility.

This module provides the AnalysisLogger class for capturing comprehensive
metadata about analysis execution to ensure reproducibility.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class AnalysisLogger:
    """Logger for capturing analysis metadata to ensure reproducibility.
    
    This class captures:
    - Data sources (API endpoints, parameters, timestamps)
    - Statistical methods (test implementations, versions, assumptions)
    - Execution environment (commit hashes, dependencies, seeds)
    
    All logs are stored in structured JSON format.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize the AnalysisLogger.
        
        Args:
            log_dir: Directory to store log files. If None, uses './logs'
        """
        self.log_dir = Path(log_dir) if log_dir else Path('./logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata: Dict[str, Any] = {
            'analysis_id': self._generate_analysis_id(),
            'created_at': datetime.utcnow().isoformat(),
            'data_sources': [],
            'statistical_methods': [],
            'execution_environment': {},
        }
    
    def _generate_analysis_id(self) -> str:
        """Generate a unique analysis ID based on timestamp.
        
        Returns:
            Unique analysis identifier
        """
        return f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def log_data_sources(
        self,
        source_name: str,
        endpoint: str,
        parameters: Dict[str, Any],
        timestamp: Optional[str] = None,
        data_version: Optional[str] = None,
        record_count: Optional[int] = None,
    ) -> None:
        """Log data source information for reproducibility.
        
        Args:
            source_name: Name of the data source (e.g., 'wikimedia_pageviews')
            endpoint: API endpoint or data location
            parameters: Query parameters or filters applied
            timestamp: Timestamp of data acquisition (ISO format)
            data_version: Version identifier for the data
            record_count: Number of records acquired
        """
        source_info = {
            'source_name': source_name,
            'endpoint': endpoint,
            'parameters': self._serialize_parameters(parameters),
            'timestamp': timestamp or datetime.utcnow().isoformat(),
            'data_version': data_version,
            'record_count': record_count,
        }
        
        self.metadata['data_sources'].append(source_info)
    
    def log_statistical_methods(
        self,
        method_name: str,
        implementation: str,
        library_version: str,
        assumptions: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        significance_level: Optional[float] = None,
    ) -> None:
        """Log statistical method information for reproducibility.
        
        Args:
            method_name: Name of the statistical method (e.g., 't_test', 'ARIMA')
            implementation: Library and function used (e.g., 'scipy.stats.ttest_ind')
            library_version: Version of the library used
            assumptions: List of statistical assumptions made
            parameters: Method-specific parameters
            significance_level: Significance level (alpha) if applicable
        """
        method_info = {
            'method_name': method_name,
            'implementation': implementation,
            'library_version': library_version,
            'assumptions': assumptions,
            'parameters': self._serialize_parameters(parameters) if parameters else {},
            'significance_level': significance_level,
            'logged_at': datetime.utcnow().isoformat(),
        }
        
        self.metadata['statistical_methods'].append(method_info)
    
    def log_execution_environment(
        self,
        random_seed: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log execution environment information for reproducibility.
        
        Args:
            random_seed: Random seed used for reproducibility
            additional_info: Additional environment information
        """
        env_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'commit_hash': self._get_git_commit_hash(),
            'dependencies': self._get_dependency_versions(),
            'random_seed': random_seed,
            'numpy_random_state': self._get_numpy_random_state(),
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        if additional_info:
            env_info.update(additional_info)
        
        self.metadata['execution_environment'] = env_info
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """Get the current git commit hash.
        
        Returns:
            Git commit hash or None if not in a git repository
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None
    
    def _get_dependency_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies.
        
        Returns:
            Dictionary mapping package names to versions
        """
        dependencies = {}
        
        packages = [
            'pandas',
            'numpy',
            'scipy',
            'statsmodels',
            'scikit-learn',
            'prophet',
            'pmdarima',
            'matplotlib',
            'seaborn',
            'plotly',
            'ruptures',
        ]
        
        for package in packages:
            try:
                if package == 'scikit-learn':
                    import sklearn
                    dependencies[package] = sklearn.__version__
                else:
                    module = __import__(package)
                    dependencies[package] = getattr(module, '__version__', 'unknown')
            except ImportError:
                dependencies[package] = 'not installed'
        
        return dependencies
    
    def _get_numpy_random_state(self) -> Optional[str]:
        """Get the current numpy random state.
        
        Returns:
            String representation of numpy random state or None
        """
        try:
            state = np.random.get_state()
            # Convert to serializable format (just the seed info)
            return f"MT19937_{state[1][0]}"
        except Exception:
            return None
    
    def _serialize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize parameters to JSON-compatible format.
        
        Args:
            parameters: Parameters dictionary
            
        Returns:
            JSON-serializable parameters dictionary
        """
        serialized = {}
        
        for key, value in parameters.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, tuple)):
                serialized[key] = [self._serialize_value(v) for v in value]
            elif isinstance(value, dict):
                serialized[key] = self._serialize_parameters(value)
            elif isinstance(value, (pd.Timestamp, datetime)):
                serialized[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value to JSON-compatible format.
        
        Args:
            value: Value to serialize
            
        Returns:
            JSON-serializable value
        """
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return str(value)
    
    def save_log(self, filename: Optional[str] = None) -> Path:
        """Save the log to a JSON file.
        
        Args:
            filename: Optional filename. If None, uses analysis_id
            
        Returns:
            Path to the saved log file
        """
        if filename is None:
            filename = f"{self.metadata['analysis_id']}.json"
        
        log_path = self.log_dir / filename
        
        with open(log_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return log_path
    
    def load_log(self, log_path: Path) -> Dict[str, Any]:
        """Load a log from a JSON file.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Loaded metadata dictionary
        """
        with open(log_path, 'r') as f:
            return json.load(f)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get the current metadata dictionary.
        
        Returns:
            Complete metadata dictionary
        """
        return self.metadata.copy()
    
    def add_custom_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the log.
        
        Args:
            key: Metadata key
            value: Metadata value (must be JSON-serializable)
        """
        self.metadata[key] = self._serialize_value(value)
