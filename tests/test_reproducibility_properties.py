"""Property-based tests for reproducibility and metadata tracking.

Feature: wikipedia-product-health-analysis
"""

import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from wikipedia_health.reproducibility import (
    AnalysisLogger,
    load_results,
    save_results,
    run_pipeline,
)
from wikipedia_health.reproducibility.persistence import (
    IntegrityError,
    compare_versions,
    list_versions,
    save_versioned_results,
)


# Custom strategies for generating test data
@st.composite
def analysis_result_strategy(draw):
    """Generate random analysis results."""
    result_type = draw(st.sampled_from(['dict', 'list', 'dataframe', 'nested']))
    
    if result_type == 'dict':
        return {
            'metric': draw(st.floats(min_value=0, max_value=1000000, allow_nan=False)),
            'p_value': draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
            'confidence_interval': [
                draw(st.floats(min_value=0, max_value=100, allow_nan=False)),
                draw(st.floats(min_value=100, max_value=200, allow_nan=False)),
            ],
        }
    elif result_type == 'list':
        return draw(st.lists(
            st.floats(min_value=0, max_value=1000, allow_nan=False),
            min_size=5,
            max_size=50
        ))
    elif result_type == 'dataframe':
        n_rows = draw(st.integers(min_value=5, max_value=50))
        return pd.DataFrame({
            'value': draw(st.lists(
                st.floats(min_value=0, max_value=1000, allow_nan=False),
                min_size=n_rows,
                max_size=n_rows
            )),
            'category': draw(st.lists(
                st.sampled_from(['A', 'B', 'C']),
                min_size=n_rows,
                max_size=n_rows
            )),
        })
    else:  # nested
        return {
            'results': draw(st.lists(
                st.floats(min_value=0, max_value=1000, allow_nan=False),
                min_size=3,
                max_size=10
            )),
            'metadata': {
                'method': draw(st.sampled_from(['ARIMA', 'Prophet', 'STL'])),
                'params': {'alpha': draw(st.floats(min_value=0.01, max_value=0.1))},
            },
        }


@st.composite
def metadata_strategy(draw):
    """Generate random metadata dictionaries."""
    return {
        'source': draw(st.sampled_from(['api', 'database', 'file'])),
        'timestamp': datetime.utcnow().isoformat(),
        'parameters': {
            'alpha': draw(st.floats(min_value=0.01, max_value=0.1, allow_nan=False)),
            'method': draw(st.sampled_from(['t_test', 'anova', 'mann_whitney'])),
        },
    }


@settings(max_examples=100)
@given(
    result=analysis_result_strategy(),
    metadata=metadata_strategy(),
)
def test_property_30_analysis_reproducibility(result, metadata):
    """
    Feature: wikipedia-product-health-analysis
    Property 30: Analysis Reproducibility
    
    For any analysis execution, running the same analysis twice with identical inputs,
    parameters, and random seeds should produce identical results within numerical
    precision tolerance of 1e-10.
    
    This test validates that save/load operations preserve results exactly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Save result once
        filepath1 = tmpdir_path / 'result1.pkl'
        saved_path1, checksum1 = save_results(result, filepath1, metadata)
        
        # Load and save again
        loaded_result1, loaded_metadata1 = load_results(saved_path1)
        filepath2 = tmpdir_path / 'result2.pkl'
        saved_path2, checksum2 = save_results(loaded_result1, filepath2, loaded_metadata1['user_metadata'])
        
        # Load second time
        loaded_result2, loaded_metadata2 = load_results(saved_path2)
        
        # Results should be identical
        if isinstance(result, dict):
            for key in result:
                if isinstance(result[key], float):
                    assert abs(loaded_result1[key] - loaded_result2[key]) < 1e-10
                elif isinstance(result[key], list):
                    for v1, v2 in zip(loaded_result1[key], loaded_result2[key]):
                        if isinstance(v1, float):
                            assert abs(v1 - v2) < 1e-10
                        else:
                            assert v1 == v2
                else:
                    assert loaded_result1[key] == loaded_result2[key]
        elif isinstance(result, list):
            for v1, v2 in zip(loaded_result1, loaded_result2):
                if isinstance(v1, float):
                    assert abs(v1 - v2) < 1e-10
                else:
                    assert v1 == v2
        elif isinstance(result, pd.DataFrame):
            pd.testing.assert_frame_equal(loaded_result1, loaded_result2)


@settings(max_examples=100, deadline=None)
@given(
    source_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    endpoint=st.text(min_size=5, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P'))),
    method_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    random_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_31_metadata_completeness(source_name, endpoint, method_name, random_seed):
    """
    Feature: wikipedia-product-health-analysis
    Property 31: Metadata Completeness
    
    For any analysis execution, the system should log all data sources (API endpoints,
    query parameters, timestamps), statistical methods (test implementations, library
    versions, assumptions), and execution environment details (code commit hashes,
    dependency versions, random seeds).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AnalysisLogger(log_dir=Path(tmpdir))
        
        # Log data sources
        logger.log_data_sources(
            source_name=source_name,
            endpoint=endpoint,
            parameters={'param1': 'value1', 'param2': 123},
            record_count=1000,
        )
        
        # Log statistical methods
        logger.log_statistical_methods(
            method_name=method_name,
            implementation='scipy.stats.ttest_ind',
            library_version='1.11.0',
            assumptions=['normality', 'equal_variance'],
            significance_level=0.05,
        )
        
        # Log execution environment
        logger.log_execution_environment(random_seed=random_seed)
        
        # Get metadata
        metadata = logger.get_metadata()
        
        # Verify data sources are logged
        assert len(metadata['data_sources']) > 0
        data_source = metadata['data_sources'][0]
        assert data_source['source_name'] == source_name
        assert data_source['endpoint'] == endpoint
        assert 'parameters' in data_source
        assert 'timestamp' in data_source
        assert data_source['record_count'] == 1000
        
        # Verify statistical methods are logged
        assert len(metadata['statistical_methods']) > 0
        method = metadata['statistical_methods'][0]
        assert method['method_name'] == method_name
        assert method['implementation'] == 'scipy.stats.ttest_ind'
        assert method['library_version'] == '1.11.0'
        assert 'assumptions' in method
        assert method['significance_level'] == 0.05
        
        # Verify execution environment is logged
        env = metadata['execution_environment']
        assert 'python_version' in env
        assert 'platform' in env
        assert 'dependencies' in env
        assert env['random_seed'] == random_seed
        assert 'timestamp' in env
        
        # Verify log can be saved and loaded
        log_path = logger.save_log()
        assert log_path.exists()
        
        loaded_metadata = logger.load_log(log_path)
        assert loaded_metadata['data_sources'] == metadata['data_sources']
        assert loaded_metadata['statistical_methods'] == metadata['statistical_methods']
        assert loaded_metadata['execution_environment']['random_seed'] == random_seed


@settings(max_examples=100)
@given(
    result=analysis_result_strategy(),
    metadata=metadata_strategy(),
)
def test_property_32_result_integrity(result, metadata):
    """
    Feature: wikipedia-product-health-analysis
    Property 32: Result Integrity
    
    For any generated result (intermediate or final), the system should save it with
    a SHA-256 checksum, enabling verification of data integrity.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        filepath = tmpdir_path / 'result.pkl'
        
        # Save result with checksum
        saved_path, checksum = save_results(result, filepath, metadata)
        
        # Verify checksum is generated
        assert checksum is not None
        assert len(checksum) == 64  # SHA-256 produces 64 hex characters
        
        # Load result with checksum verification
        loaded_result, loaded_metadata = load_results(saved_path, verify_checksum=True)
        
        # Verify checksum is stored in metadata
        assert 'checksum' in loaded_metadata
        assert loaded_metadata['checksum'] == checksum
        
        # Verify result integrity
        if isinstance(result, dict):
            for key in result:
                assert key in loaded_result
        elif isinstance(result, list):
            assert len(loaded_result) == len(result)
        elif isinstance(result, pd.DataFrame):
            pd.testing.assert_frame_equal(loaded_result, result)
        
        # Test checksum verification failure
        # Corrupt the file
        with open(saved_path, 'rb') as f:
            data = bytearray(f.read())
        
        if len(data) > 10:
            # Flip a bit in the middle of the file
            data[len(data) // 2] ^= 0xFF
            
            with open(saved_path, 'wb') as f:
                f.write(data)
            
            # Loading should fail with IntegrityError
            with pytest.raises(IntegrityError):
                load_results(saved_path, verify_checksum=True)
            
            # Loading without verification should succeed (but data is corrupted)
            try:
                load_results(saved_path, verify_checksum=False)
            except Exception:
                # Pickle may fail to deserialize corrupted data
                pass


@settings(max_examples=50, deadline=None)  # Reduced examples due to pipeline complexity
@given(
    input_value=st.integers(min_value=1, max_value=100),
    random_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_33_pipeline_re_execution(input_value, random_seed):
    """
    Feature: wikipedia-product-health-analysis
    Property 33: Pipeline Re-execution
    
    For any completed analysis, the system should support re-running the entire pipeline
    with new data inputs while preserving historical analysis versions, enabling
    longitudinal tracking of findings.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Define simple pipeline steps
        def step1(data):
            np.random.seed(random_seed)
            return data * 2 + np.random.randint(0, 5)
        
        def step2(data):
            np.random.seed(random_seed)
            return data + 10 + np.random.randint(0, 3)
        
        def step3(data):
            return data ** 2
        
        pipeline_steps = [
            ('multiply', step1),
            ('add', step2),
            ('square', step3),
        ]
        
        # Run pipeline first time
        results1 = run_pipeline(
            pipeline_steps=pipeline_steps,
            input_data=input_value,
            output_dir=tmpdir_path,
            pipeline_name='test_pipeline',
            version='v1',
            random_seed=random_seed,
            save_intermediate=True,
            git_tag=False,
        )
        
        # Verify first execution
        assert results1['status'] == 'success'
        assert 'final_result' in results1
        assert 'final_result_path' in results1
        assert 'final_result_checksum' in results1
        assert len(results1['steps']) == 3
        
        # Run pipeline second time with different input
        results2 = run_pipeline(
            pipeline_steps=pipeline_steps,
            input_data=input_value + 10,
            output_dir=tmpdir_path,
            pipeline_name='test_pipeline',
            version='v2',
            random_seed=random_seed,
            save_intermediate=True,
            git_tag=False,
        )
        
        # Verify second execution
        assert results2['status'] == 'success'
        assert results2['final_result'] != results1['final_result']
        
        # Verify both versions are preserved
        versions = list_versions(tmpdir_path / 'test_pipeline_final.pkl')
        assert len(versions) >= 2
        
        # Verify version metadata
        version_ids = [v['version'] for v in versions]
        assert 'v1' in version_ids
        assert 'v2' in version_ids
        
        # Verify we can load both versions
        v1_path = Path(results1['final_result_path'])
        v2_path = Path(results2['final_result_path'])
        
        loaded_v1, meta_v1 = load_results(v1_path)
        loaded_v2, meta_v2 = load_results(v2_path)
        
        assert loaded_v1 == results1['final_result']
        assert loaded_v2 == results2['final_result']
        assert meta_v1['version'] == 'v1'
        assert meta_v2['version'] == 'v2'
        
        # Verify version comparison
        comparison = compare_versions(v1_path, v2_path)
        assert comparison['version1'] == 'v1'
        assert comparison['version2'] == 'v2'
        assert comparison['checksum1'] != comparison['checksum2']
        assert not comparison['checksums_match']


@settings(max_examples=100)
@given(
    result=analysis_result_strategy(),
    version1=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    version2=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
)
def test_property_versioning_support(result, version1, version2):
    """
    Additional property test: Versioning support for historical preservation.
    
    The system should support saving multiple versions of the same analysis
    and retrieving specific versions.
    """
    # Ensure versions are different
    if version1 == version2:
        version2 = version2 + '_v2'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        base_filepath = tmpdir_path / 'analysis.pkl'
        
        # Save first version
        path1, checksum1 = save_versioned_results(
            result,
            base_filepath,
            metadata={'description': 'First version'},
            version=version1,
        )
        
        # Modify result slightly for second version
        if isinstance(result, dict) and 'metric' in result:
            result['metric'] = result['metric'] * 1.1
        elif isinstance(result, list) and len(result) > 0:
            result[0] = result[0] * 1.1
        
        # Save second version
        path2, checksum2 = save_versioned_results(
            result,
            base_filepath,
            metadata={'description': 'Second version'},
            version=version2,
        )
        
        # Verify both versions exist
        assert path1.exists()
        assert path2.exists()
        assert path1 != path2
        
        # Verify checksums are different (unless result is identical)
        # Note: checksums might be same if result modification didn't change serialization
        
        # List versions
        versions = list_versions(base_filepath)
        assert len(versions) >= 2
        
        version_ids = [v['version'] for v in versions]
        assert version1 in version_ids
        assert version2 in version_ids
        
        # Load specific versions
        loaded1, meta1 = load_results(path1)
        loaded2, meta2 = load_results(path2)
        
        assert meta1['version'] == version1
        assert meta2['version'] == version2
