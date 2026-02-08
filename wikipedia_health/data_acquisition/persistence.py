"""Data persistence layer for saving and loading time series data with integrity checks."""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import pandas as pd

from wikipedia_health.models.data_models import TimeSeriesData


logger = logging.getLogger(__name__)


def _compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum for data integrity.
    
    Args:
        data: Bytes to compute checksum for
        
    Returns:
        Hexadecimal checksum string
    """
    return hashlib.sha256(data).hexdigest()


def save_timeseries_data(
    data: TimeSeriesData,
    filepath: str,
    version: Optional[str] = None
) -> Tuple[str, str]:
    """Save TimeSeriesData with metadata and checksum.
    
    Args:
        data: TimeSeriesData object to save
        filepath: Path to save data (without extension)
        version: Optional version identifier for historical tracking
        
    Returns:
        Tuple of (data_filepath, checksum)
    """
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add version to filename if provided
    if version:
        base_name = filepath.stem
        data_filepath = filepath.parent / f"{base_name}_v{version}.pkl"
        metadata_filepath = filepath.parent / f"{base_name}_v{version}_metadata.json"
    else:
        data_filepath = filepath.with_suffix('.pkl')
        metadata_filepath = filepath.with_suffix('_metadata.json')
    
    # Serialize data
    data_bytes = pickle.dumps(data)
    
    # Compute checksum
    checksum = _compute_checksum(data_bytes)
    
    # Save data
    with open(data_filepath, 'wb') as f:
        f.write(data_bytes)
    
    # Prepare metadata
    metadata = {
        'platform': data.platform,
        'metric_type': data.metric_type,
        'num_records': len(data.values),
        'date_range': {
            'start': str(data.date.min()),
            'end': str(data.date.max())
        },
        'checksum': checksum,
        'version': version,
        'saved_at': datetime.now().isoformat(),
        'user_metadata': data.metadata
    }
    
    # Save metadata
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(
        f"Saved TimeSeriesData to {data_filepath} "
        f"(checksum: {checksum[:16]}..., {len(data.values)} records)"
    )
    
    return str(data_filepath), checksum


def load_timeseries_data(
    filepath: str,
    verify_checksum: bool = True
) -> TimeSeriesData:
    """Load TimeSeriesData with optional checksum verification.
    
    Args:
        filepath: Path to data file (with or without .pkl extension)
        verify_checksum: Whether to verify data integrity with checksum
        
    Returns:
        TimeSeriesData object
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If checksum verification fails
    """
    filepath = Path(filepath)
    
    # Ensure .pkl extension
    if filepath.suffix != '.pkl':
        filepath = filepath.with_suffix('.pkl')
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load data
    with open(filepath, 'rb') as f:
        data_bytes = f.read()
    
    # Verify checksum if requested
    if verify_checksum:
        # Load metadata
        metadata_filepath = filepath.with_suffix('').with_suffix('_metadata.json')
        
        if metadata_filepath.exists():
            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)
            
            expected_checksum = metadata.get('checksum')
            if expected_checksum:
                actual_checksum = _compute_checksum(data_bytes)
                
                if actual_checksum != expected_checksum:
                    raise ValueError(
                        f"Checksum verification failed for {filepath}. "
                        f"Expected: {expected_checksum[:16]}..., "
                        f"Got: {actual_checksum[:16]}..."
                    )
                
                logger.info(f"Checksum verification passed for {filepath}")
        else:
            logger.warning(f"Metadata file not found: {metadata_filepath}. Skipping checksum verification.")
    
    # Deserialize data
    data = pickle.loads(data_bytes)
    
    logger.info(f"Loaded TimeSeriesData from {filepath} ({len(data.values)} records)")
    
    return data


def list_versions(filepath: str) -> list:
    """List all available versions of a saved dataset.
    
    Args:
        filepath: Base filepath (without version suffix)
        
    Returns:
        List of version identifiers found
    """
    filepath = Path(filepath)
    base_name = filepath.stem
    directory = filepath.parent
    
    if not directory.exists():
        return []
    
    # Find all versioned files
    pattern = f"{base_name}_v*.pkl"
    versioned_files = list(directory.glob(pattern))
    
    # Extract version identifiers
    versions = []
    for file in versioned_files:
        # Extract version from filename (format: basename_vVERSION.pkl)
        name = file.stem
        if '_v' in name:
            version = name.split('_v')[-1]
            versions.append(version)
    
    versions.sort()
    
    logger.info(f"Found {len(versions)} versions for {base_name}")
    
    return versions


def get_metadata(filepath: str) -> Optional[Dict]:
    """Load metadata for a saved dataset.
    
    Args:
        filepath: Path to data file (with or without extension)
        
    Returns:
        Metadata dictionary or None if not found
    """
    filepath = Path(filepath)
    
    # Handle .pkl extension
    if filepath.suffix == '.pkl':
        metadata_filepath = filepath.with_suffix('').with_suffix('_metadata.json')
    else:
        metadata_filepath = filepath.with_suffix('_metadata.json')
    
    if not metadata_filepath.exists():
        logger.warning(f"Metadata file not found: {metadata_filepath}")
        return None
    
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    
    return metadata
