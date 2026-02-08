"""Pipeline Re-execution Support.

This module provides functions for orchestrating full analysis workflows,
version comparison for longitudinal tracking, and git integration for
tagging analysis versions.
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from wikipedia_health.reproducibility.logger import AnalysisLogger
from wikipedia_health.reproducibility.persistence import (
    compare_versions,
    list_versions,
    save_versioned_results,
)


class PipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


class GitIntegrationError(Exception):
    """Raised when git operations fail."""
    pass


def run_pipeline(
    pipeline_steps: List[Tuple[str, Callable]],
    input_data: Any,
    output_dir: Path,
    pipeline_name: str = "analysis_pipeline",
    version: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_intermediate: bool = True,
    git_tag: bool = False,
) -> Dict[str, Any]:
    """Execute a full analysis pipeline with reproducibility tracking.
    
    This function orchestrates a complete analysis workflow:
    - Executes each pipeline step in sequence
    - Logs all metadata for reproducibility
    - Saves intermediate and final results with checksums
    - Optionally creates git tags for version tracking
    
    Args:
        pipeline_steps: List of (step_name, step_function) tuples
        input_data: Initial input data for the pipeline
        output_dir: Directory to save results
        pipeline_name: Name of the pipeline
        version: Optional version identifier
        random_seed: Random seed for reproducibility
        save_intermediate: Whether to save intermediate step results
        git_tag: Whether to create a git tag for this pipeline run
        
    Returns:
        Dictionary with pipeline results and metadata
        
    Raises:
        PipelineError: If any pipeline step fails
        
    Example:
        >>> def step1(data):
        ...     return data * 2
        >>> def step2(data):
        ...     return data + 1
        >>> steps = [('double', step1), ('increment', step2)]
        >>> results = run_pipeline(steps, 5, Path('./output'))
        >>> print(results['final_result'])  # 11
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = AnalysisLogger(log_dir=output_dir / 'logs')
    
    # Generate version if not provided
    if version is None:
        version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Log execution environment
    logger.log_execution_environment(
        random_seed=random_seed,
        additional_info={
            'pipeline_name': pipeline_name,
            'pipeline_version': version,
            'num_steps': len(pipeline_steps),
            'step_names': [name for name, _ in pipeline_steps],
        }
    )
    
    # Execute pipeline steps
    results = {
        'pipeline_name': pipeline_name,
        'version': version,
        'started_at': datetime.utcnow().isoformat(),
        'steps': {},
        'intermediate_results': {},
    }
    
    current_data = input_data
    
    for step_idx, (step_name, step_function) in enumerate(pipeline_steps):
        step_start = datetime.utcnow()
        
        try:
            # Execute step
            step_result = step_function(current_data)
            
            # Record step execution
            step_info = {
                'step_index': step_idx,
                'step_name': step_name,
                'started_at': step_start.isoformat(),
                'completed_at': datetime.utcnow().isoformat(),
                'duration_seconds': (datetime.utcnow() - step_start).total_seconds(),
                'status': 'success',
            }
            
            results['steps'][step_name] = step_info
            
            # Save intermediate result if requested
            if save_intermediate:
                intermediate_path = output_dir / f"{pipeline_name}_{step_name}_{version}.pkl"
                saved_path, _ = save_versioned_results(
                    step_result,
                    intermediate_path,
                    metadata={
                        'pipeline_name': pipeline_name,
                        'step_name': step_name,
                        'step_index': step_idx,
                    },
                    version=version,
                )
                results['intermediate_results'][step_name] = str(saved_path)
            
            # Update current data for next step
            current_data = step_result
            
        except Exception as e:
            # Record failure
            step_info = {
                'step_index': step_idx,
                'step_name': step_name,
                'started_at': step_start.isoformat(),
                'failed_at': datetime.utcnow().isoformat(),
                'duration_seconds': (datetime.utcnow() - step_start).total_seconds(),
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
            }
            
            results['steps'][step_name] = step_info
            results['status'] = 'failed'
            results['failed_step'] = step_name
            
            # Save partial results
            logger.add_custom_metadata('pipeline_results', results)
            logger.save_log()
            
            raise PipelineError(
                f"Pipeline failed at step '{step_name}': {e}"
            ) from e
    
    # Save final result
    final_path = output_dir / f"{pipeline_name}_final_{version}.pkl"
    final_path_saved, checksum = save_versioned_results(
        current_data,
        final_path,
        metadata={
            'pipeline_name': pipeline_name,
            'version': version,
            'num_steps': len(pipeline_steps),
        },
        version=version,
    )
    
    results['final_result'] = current_data
    results['final_result_path'] = str(final_path_saved)
    results['final_result_checksum'] = checksum
    results['completed_at'] = datetime.utcnow().isoformat()
    results['status'] = 'success'
    
    # Save pipeline metadata
    logger.add_custom_metadata('pipeline_results', results)
    log_path = logger.save_log()
    results['log_path'] = str(log_path)
    
    # Create git tag if requested
    if git_tag:
        try:
            tag_name = f"{pipeline_name}_{version}"
            create_git_tag(
                tag_name=tag_name,
                message=f"Pipeline execution: {pipeline_name} version {version}",
            )
            results['git_tag'] = tag_name
        except GitIntegrationError as e:
            results['git_tag_error'] = str(e)
    
    return results


def compare_pipeline_versions(
    pipeline_name: str,
    output_dir: Path,
    version1: Optional[str] = None,
    version2: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two versions of a pipeline execution for longitudinal tracking.
    
    This function compares results from different pipeline executions to
    track how analyses change over time.
    
    Args:
        pipeline_name: Name of the pipeline
        output_dir: Directory containing pipeline results
        version1: First version to compare (None = latest)
        version2: Second version to compare (None = second latest)
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> comparison = compare_pipeline_versions(
        ...     'trend_analysis',
        ...     Path('./output'),
        ...     version1='20240101_120000',
        ...     version2='20240201_120000'
        ... )
        >>> print(comparison['results_match'])
    """
    output_dir = Path(output_dir)
    
    # Find available versions
    base_path = output_dir / f"{pipeline_name}_final.pkl"
    versions = list_versions(base_path)
    
    if len(versions) < 2:
        raise ValueError(
            f"Need at least 2 versions to compare, found {len(versions)}"
        )
    
    # Select versions to compare
    if version1 is None:
        version1_meta = versions[0]
    else:
        version1_meta = next(
            (v for v in versions if v['version'] == version1),
            None
        )
        if version1_meta is None:
            raise ValueError(f"Version {version1} not found")
    
    if version2 is None:
        version2_meta = versions[1] if version1 is None else versions[0]
    else:
        version2_meta = next(
            (v for v in versions if v['version'] == version2),
            None
        )
        if version2_meta is None:
            raise ValueError(f"Version {version2} not found")
    
    # Compare versions
    comparison = compare_versions(
        version1_meta['filepath'],
        version2_meta['filepath'],
    )
    
    comparison['pipeline_name'] = pipeline_name
    comparison['comparison_timestamp'] = datetime.utcnow().isoformat()
    
    return comparison


def create_git_tag(
    tag_name: str,
    message: Optional[str] = None,
    annotated: bool = True,
) -> str:
    """Create a git tag for analysis version tracking.
    
    Args:
        tag_name: Name of the tag
        message: Optional tag message (required for annotated tags)
        annotated: Whether to create an annotated tag
        
    Returns:
        Tag name that was created
        
    Raises:
        GitIntegrationError: If git operations fail
        
    Example:
        >>> tag = create_git_tag(
        ...     'analysis_v1.0',
        ...     message='First analysis version'
        ... )
        >>> print(f"Created tag: {tag}")
    """
    try:
        # Check if we're in a git repository
        subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            capture_output=True,
            check=True,
            timeout=5,
        )
        
        # Create tag
        if annotated:
            if message is None:
                message = f"Analysis tag: {tag_name}"
            
            subprocess.run(
                ['git', 'tag', '-a', tag_name, '-m', message],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        else:
            subprocess.run(
                ['git', 'tag', tag_name],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        
        return tag_name
        
    except subprocess.CalledProcessError as e:
        raise GitIntegrationError(
            f"Failed to create git tag '{tag_name}': {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired:
        raise GitIntegrationError(
            f"Git command timed out while creating tag '{tag_name}'"
        )
    except FileNotFoundError:
        raise GitIntegrationError(
            "Git command not found. Is git installed?"
        )


def list_git_tags(pattern: Optional[str] = None) -> List[str]:
    """List git tags, optionally filtered by pattern.
    
    Args:
        pattern: Optional pattern to filter tags (e.g., 'analysis_*')
        
    Returns:
        List of tag names
        
    Raises:
        GitIntegrationError: If git operations fail
        
    Example:
        >>> tags = list_git_tags(pattern='analysis_*')
        >>> for tag in tags:
        ...     print(tag)
    """
    try:
        cmd = ['git', 'tag', '-l']
        if pattern:
            cmd.append(pattern)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        
        tags = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        return tags
        
    except subprocess.CalledProcessError as e:
        raise GitIntegrationError(
            f"Failed to list git tags: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired:
        raise GitIntegrationError("Git command timed out while listing tags")
    except FileNotFoundError:
        raise GitIntegrationError("Git command not found. Is git installed?")


def get_pipeline_history(
    pipeline_name: str,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Get execution history for a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        output_dir: Directory containing pipeline results
        
    Returns:
        List of execution metadata sorted by version (newest first)
        
    Example:
        >>> history = get_pipeline_history('trend_analysis', Path('./output'))
        >>> for execution in history:
        ...     print(f"{execution['version']}: {execution['saved_at']}")
    """
    output_dir = Path(output_dir)
    base_path = output_dir / f"{pipeline_name}_final.pkl"
    
    return list_versions(base_path)
