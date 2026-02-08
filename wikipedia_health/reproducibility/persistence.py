"""Result Persistence with Integrity Checks.

This module provides functions for saving and loading analysis results
with SHA-256 checksum generation and verification, plus versioning support
for historical analysis preservation.
"""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


class IntegrityError(Exception):
    """Raised when checksum verification fails."""
    pass


def _compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum of data.
    
    Args:
        data: Bytes to compute checksum for
        
    Returns:
        Hexadecimal SHA-256 checksum
    """
    return hashlib.sha256(data).hexdigest()


def _serialize_result(result: Any) -> bytes:
    """Serialize result to bytes using pickle.
    
    Args:
        result: Result object to serialize
        
    Returns:
        Serialized bytes
    """
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_result(data: bytes) -> Any:
    """Deserialize result from bytes using pickle.
    
    Args:
        data: Serialized bytes
        
    Returns:
        Deserialized result object
    """
    return pickle.loads(data)


def save_results(
    result: Any,
    filepath: Path,
    metadata: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
) -> Tuple[Path, str]:
    """Save analysis results with SHA-256 checksum generation.
    
    This function saves results in a versioned format with integrity checks:
    - Serializes the result using pickle
    - Computes SHA-256 checksum
    - Saves result data and metadata with checksum
    - Supports versioning for historical preservation
    
    Args:
        result: Analysis result to save (any picklable object)
        filepath: Path to save the result
        metadata: Optional metadata dictionary
        version: Optional version identifier (defaults to timestamp)
        
    Returns:
        Tuple of (saved_path, checksum)
        
    Example:
        >>> result = {'analysis': 'data'}
        >>> path, checksum = save_results(result, Path('results.pkl'))
        >>> print(f"Saved with checksum: {checksum}")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate version if not provided
    if version is None:
        version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Serialize the result
    result_data = _serialize_result(result)
    
    # Compute checksum
    checksum = _compute_checksum(result_data)
    
    # Prepare metadata
    save_metadata = {
        'version': version,
        'checksum': checksum,
        'saved_at': datetime.utcnow().isoformat(),
        'size_bytes': len(result_data),
    }
    
    if metadata:
        save_metadata['user_metadata'] = metadata
    
    # Save result data
    with open(filepath, 'wb') as f:
        f.write(result_data)
    
    # Save metadata alongside result
    metadata_path = filepath.with_suffix(filepath.suffix + '.meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(save_metadata, f, indent=2)
    
    return filepath, checksum


def load_results(
    filepath: Path,
    verify_checksum: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Load analysis results with checksum verification.
    
    This function loads results and verifies integrity:
    - Loads result data and metadata
    - Recomputes checksum and verifies against stored checksum
    - Raises IntegrityError if verification fails
    - Returns result and metadata
    
    Args:
        filepath: Path to the saved result
        verify_checksum: Whether to verify checksum (default: True)
        
    Returns:
        Tuple of (result, metadata)
        
    Raises:
        IntegrityError: If checksum verification fails
        FileNotFoundError: If result or metadata file not found
        
    Example:
        >>> result, metadata = load_results(Path('results.pkl'))
        >>> print(f"Loaded version: {metadata['version']}")
    """
    filepath = Path(filepath)
    
    # Load result data
    with open(filepath, 'rb') as f:
        result_data = f.read()
    
    # Load metadata
    metadata_path = filepath.with_suffix(filepath.suffix + '.meta.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Verify checksum if requested
    if verify_checksum:
        stored_checksum = metadata.get('checksum')
        if not stored_checksum:
            raise IntegrityError("No checksum found in metadata")
        
        computed_checksum = _compute_checksum(result_data)
        if computed_checksum != stored_checksum:
            raise IntegrityError(
                f"Checksum mismatch: expected {stored_checksum}, "
                f"got {computed_checksum}"
            )
    
    # Deserialize result
    result = _deserialize_result(result_data)
    
    return result, metadata


def list_versions(base_filepath: Path) -> list[Dict[str, Any]]:
    """List all versions of a saved result.
    
    This function finds all versioned files matching the base filepath pattern
    and returns their metadata sorted by version.
    
    Args:
        base_filepath: Base filepath pattern (e.g., 'results.pkl')
        
    Returns:
        List of metadata dictionaries sorted by version (newest first)
        
    Example:
        >>> versions = list_versions(Path('results.pkl'))
        >>> for v in versions:
        ...     print(f"Version {v['version']}: {v['saved_at']}")
    """
    base_filepath = Path(base_filepath)
    parent_dir = base_filepath.parent
    base_name = base_filepath.stem
    
    if not parent_dir.exists():
        return []
    
    # Find all versioned files
    versions = []
    for meta_file in parent_dir.glob(f"{base_name}*.meta.json"):
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                # Remove .meta.json to get the actual result file path
                # meta_file is like: result_v1.pkl.meta.json
                # We want: result_v1.pkl
                result_file = str(meta_file).replace('.meta.json', '')
                metadata['filepath'] = Path(result_file)
                versions.append(metadata)
        except (json.JSONDecodeError, IOError):
            continue
    
    # Sort by version (newest first)
    versions.sort(key=lambda x: x.get('version', ''), reverse=True)
    
    return versions


def save_versioned_results(
    result: Any,
    base_filepath: Path,
    metadata: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
) -> Tuple[Path, str]:
    """Save results with automatic versioning.
    
    This function saves results with a version suffix in the filename,
    preserving historical versions.
    
    Args:
        result: Analysis result to save
        base_filepath: Base filepath (version will be appended)
        metadata: Optional metadata dictionary
        version: Optional version identifier (defaults to timestamp)
        
    Returns:
        Tuple of (saved_path, checksum)
        
    Example:
        >>> result = {'analysis': 'data'}
        >>> path, checksum = save_versioned_results(
        ...     result, Path('results.pkl'), version='v1.0'
        ... )
        >>> # Saves to 'results_v1.0.pkl'
    """
    base_filepath = Path(base_filepath)
    
    # Generate version if not provided
    if version is None:
        version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Create versioned filepath
    versioned_filepath = base_filepath.with_stem(
        f"{base_filepath.stem}_{version}"
    )
    
    # Save with the specified version (don't let save_results generate another one)
    return save_results(result, versioned_filepath, metadata, version=version)


def load_latest_version(base_filepath: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load the latest version of a saved result.
    
    Args:
        base_filepath: Base filepath pattern
        
    Returns:
        Tuple of (result, metadata) for the latest version
        
    Raises:
        FileNotFoundError: If no versions found
        
    Example:
        >>> result, metadata = load_latest_version(Path('results.pkl'))
        >>> print(f"Loaded latest version: {metadata['version']}")
    """
    versions = list_versions(base_filepath)
    
    if not versions:
        raise FileNotFoundError(f"No versions found for {base_filepath}")
    
    latest = versions[0]
    return load_results(latest['filepath'])


def compare_versions(
    filepath1: Path,
    filepath2: Path,
) -> Dict[str, Any]:
    """Compare two versions of saved results.
    
    This function loads two versions and compares their metadata
    and checksums to identify differences.
    
    Args:
        filepath1: Path to first version
        filepath2: Path to second version
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> comparison = compare_versions(
        ...     Path('results_v1.pkl'),
        ...     Path('results_v2.pkl')
        ... )
        >>> if comparison['checksums_match']:
        ...     print("Results are identical")
    """
    # Load both versions
    result1, metadata1 = load_results(filepath1)
    result2, metadata2 = load_results(filepath2)
    
    comparison = {
        'version1': metadata1.get('version'),
        'version2': metadata2.get('version'),
        'saved_at1': metadata1.get('saved_at'),
        'saved_at2': metadata2.get('saved_at'),
        'checksum1': metadata1.get('checksum'),
        'checksum2': metadata2.get('checksum'),
        'checksums_match': metadata1.get('checksum') == metadata2.get('checksum'),
        'size_bytes1': metadata1.get('size_bytes'),
        'size_bytes2': metadata2.get('size_bytes'),
    }
    
    return comparison
