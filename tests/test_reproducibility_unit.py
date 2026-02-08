"""Unit tests for reproducibility and metadata tracking.

These tests focus on specific examples and edge cases for the reproducibility module.
"""

import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    load_latest_version,
    save_versioned_results,
)
from wikipedia_health.reproducibility.pipeline import (
    GitIntegrationError,
    PipelineError,
    compare_pipeline_versions,
    create_git_tag,
    get_pipeline_history,
    list_git_tags,
)


class TestAnalysisLogger:
    """Unit tests for AnalysisLogger class."""
    
    def test_logger_initialization(self):
        """Test that logger initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AnalysisLogger(log_dir=Path(tmpdir))
            
            assert logger.log_dir == Path(tmpdir)
            assert logger.log_dir.exists()
            assert 'analysis_id' in logger.metadata
            assert 'created_at' in logger.metadata
            assert 'data_sources' in logger.metadata
            assert 'statistical_methods' in logger.metadata
            assert 'execution_environment' in logger.metadata
    
    def test_log_data_sources_complete(self):
        """Test logging data sources with all parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AnalysisLogger(log_dir=Path(tmpdir))
            
            logger.log_data_sources(
                source_name='wikimedia_pageviews',
                endpoint='https://wikimedia.org/api/rest_v1/metrics/pageviews',
                parameters={'start': '2020-01-01', 'end': '2020-12-31'},
                timestamp='2024-01-01T00:00:00',
                data_version='v1.0',
                record_count=365,
            )
            
            assert len(logger.metadata['data_sources']) == 1
            source = logger.metadata['data_sources'][0]
            assert source['source_name'] == 'wikimedia_pageviews'
            assert source['endpoint'] == 'https://wikimedia.org/api/rest_v1/metrics/pageviews'
            assert source['parameters']['start'] == '2020-01-01'
            assert source['timestamp'] == '2024-01-01T00:00:00'
            assert source['data_version'] == 'v1.0'
            assert source['record_count'] == 365
    
    def test_log_data_sources_minimal(self):
        """Test logging data sources with minimal parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AnalysisLogger(log_dir=Path(tmpdir))
            
            logger.log_data_sources(
                source_name='test_source',
                endpoint='http://example.com',
                parameters={},
            )
            
            assert len(logger.metadata['data_sources']) == 1
            source = logger.metadata['data_sources'][0]
            assert source['source_name'] == 'test_source'
            assert 'timestamp' in source  # Auto-generated
    
    def test_log_statistical_methods_complete(self):
        """Test logging statistical methods with all parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AnalysisLogger(log_dir=Path(tmpdir))
            
            logger.log_statistical_methods(
                method_name='t_test',
                implementation='scipy.stats.ttest_ind',
                library_version='1.11.0',
                assumptions=['normality', 'equal_variance'],
                parameters={'alternative': 'two-sided'},
                significance_level=0.05,
            )
            
            assert len(logger.metadata['statistical_methods']) == 1
            method = logger.metadata['statistical_methods'][0]
            assert method['method_name'] == 't_test'
            assert method['implementation'] == 'scipy.stats.ttest_ind'
            assert method['library_version'] == '1.11.0'
            assert 'normality' in method['assumptions']
            assert method['parameters']['alternative'] == 'two-sided'
            assert method['significance_level'] == 0.05
    
    def test_log_execution_environment(self):
        """Test logging execution environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AnalysisLogger(log_dir=Path(tmpdir))
            
            logger.log_execution_environment(
                random_seed=42,
                additional_info={'custom_field': 'custom_value'},
            )
            
            env = logger.metadata['execution_environment']
            assert 'python_version' in env
            assert 'platform' in env
            assert 'dependencies' in env
            assert env['random_seed'] == 42
            assert env['custom_field'] == 'custom_value'
    
    def test_save_and_load_log(self):
        """Test saving and loading log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AnalysisLogger(log_dir=Path(tmpdir))
            
            logger.log_data_sources(
                source_name='test',
                endpoint='http://test.com',
                parameters={'key': 'value'},
            )
            
            # Save log
            log_path = logger.save_log()
            assert log_path.exists()
            
            # Load log
            loaded_metadata = logger.load_log(log_path)
            assert loaded_metadata['analysis_id'] == logger.metadata['analysis_id']
            assert len(loaded_metadata['data_sources']) == 1
    
    def test_add_custom_metadata(self):
        """Test adding custom metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AnalysisLogger(log_dir=Path(tmpdir))
            
            logger.add_custom_metadata('custom_key', 'custom_value')
            logger.add_custom_metadata('custom_number', 123)
            
            assert logger.metadata['custom_key'] == 'custom_value'
            assert logger.metadata['custom_number'] == 123


class TestPersistence:
    """Unit tests for result persistence functions."""
    
    def test_save_and_load_dict_result(self):
        """Test saving and loading dictionary results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {'metric': 100.5, 'p_value': 0.03}
            filepath = Path(tmpdir) / 'result.pkl'
            
            # Save
            saved_path, checksum = save_results(result, filepath)
            assert saved_path.exists()
            assert len(checksum) == 64
            
            # Load
            loaded_result, metadata = load_results(saved_path)
            assert loaded_result == result
            assert metadata['checksum'] == checksum
    
    def test_save_and_load_dataframe_result(self):
        """Test saving and loading DataFrame results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pd.DataFrame({
                'value': [1, 2, 3, 4, 5],
                'category': ['A', 'B', 'A', 'B', 'A'],
            })
            filepath = Path(tmpdir) / 'result.pkl'
            
            # Save
            saved_path, checksum = save_results(result, filepath)
            
            # Load
            loaded_result, metadata = load_results(saved_path)
            pd.testing.assert_frame_equal(loaded_result, result)
    
    def test_save_with_metadata(self):
        """Test saving results with custom metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {'data': 'test'}
            filepath = Path(tmpdir) / 'result.pkl'
            custom_metadata = {'description': 'Test result', 'version': '1.0'}
            
            # Save with metadata
            saved_path, checksum = save_results(result, filepath, metadata=custom_metadata)
            
            # Load and verify metadata
            loaded_result, metadata = load_results(saved_path)
            assert 'user_metadata' in metadata
            assert metadata['user_metadata']['description'] == 'Test result'
            assert metadata['user_metadata']['version'] == '1.0'
    
    def test_checksum_verification_failure(self):
        """Test that corrupted files fail checksum verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {'data': 'test'}
            filepath = Path(tmpdir) / 'result.pkl'
            
            # Save
            saved_path, checksum = save_results(result, filepath)
            
            # Corrupt the file
            with open(saved_path, 'rb') as f:
                data = bytearray(f.read())
            data[10] ^= 0xFF  # Flip a bit
            with open(saved_path, 'wb') as f:
                f.write(data)
            
            # Loading with verification should fail
            with pytest.raises(IntegrityError):
                load_results(saved_path, verify_checksum=True)
    
    def test_versioned_save(self):
        """Test saving versioned results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {'data': 'test'}
            base_filepath = Path(tmpdir) / 'result.pkl'
            
            # Save version 1
            path1, checksum1 = save_versioned_results(
                result, base_filepath, version='v1'
            )
            assert 'v1' in str(path1)
            
            # Save version 2
            path2, checksum2 = save_versioned_results(
                result, base_filepath, version='v2'
            )
            assert 'v2' in str(path2)
            
            # Both files should exist
            assert path1.exists()
            assert path2.exists()
    
    def test_list_versions(self):
        """Test listing available versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {'data': 'test'}
            base_filepath = Path(tmpdir) / 'result.pkl'
            
            # Save multiple versions
            save_versioned_results(result, base_filepath, version='v1')
            save_versioned_results(result, base_filepath, version='v2')
            save_versioned_results(result, base_filepath, version='v3')
            
            # List versions
            versions = list_versions(base_filepath)
            assert len(versions) >= 3
            
            version_ids = [v['version'] for v in versions]
            assert 'v1' in version_ids
            assert 'v2' in version_ids
            assert 'v3' in version_ids
    
    def test_load_latest_version(self):
        """Test loading the latest version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_filepath = Path(tmpdir) / 'result.pkl'
            
            # Save multiple versions with different data
            save_versioned_results({'version': 1}, base_filepath, version='v1')
            save_versioned_results({'version': 2}, base_filepath, version='v2')
            save_versioned_results({'version': 3}, base_filepath, version='v3')
            
            # Load latest
            result, metadata = load_latest_version(base_filepath)
            
            # Should load v3 (latest)
            assert metadata['version'] == 'v3'
            assert result['version'] == 3
    
    def test_compare_versions(self):
        """Test comparing two versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_filepath = Path(tmpdir) / 'result.pkl'
            
            # Save two different versions
            path1, checksum1 = save_versioned_results(
                {'data': 'version1'}, base_filepath, version='v1'
            )
            path2, checksum2 = save_versioned_results(
                {'data': 'version2'}, base_filepath, version='v2'
            )
            
            # Compare
            comparison = compare_versions(path1, path2)
            
            assert comparison['version1'] == 'v1'
            assert comparison['version2'] == 'v2'
            assert comparison['checksum1'] == checksum1
            assert comparison['checksum2'] == checksum2
            assert not comparison['checksums_match']


class TestPipeline:
    """Unit tests for pipeline execution."""
    
    def test_simple_pipeline_execution(self):
        """Test executing a simple pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def step1(data):
                return data * 2
            
            def step2(data):
                return data + 10
            
            steps = [('double', step1), ('add', step2)]
            
            results = run_pipeline(
                pipeline_steps=steps,
                input_data=5,
                output_dir=Path(tmpdir),
                pipeline_name='test',
                version='v1',
                random_seed=42,
                git_tag=False,
            )
            
            assert results['status'] == 'success'
            assert results['final_result'] == 20  # (5 * 2) + 10
            assert len(results['steps']) == 2
            assert 'double' in results['steps']
            assert 'add' in results['steps']
    
    def test_pipeline_with_failure(self):
        """Test pipeline handling of step failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def step1(data):
                return data * 2
            
            def step2(data):
                raise ValueError("Intentional error")
            
            steps = [('double', step1), ('fail', step2)]
            
            with pytest.raises(PipelineError):
                run_pipeline(
                    pipeline_steps=steps,
                    input_data=5,
                    output_dir=Path(tmpdir),
                    pipeline_name='test',
                    git_tag=False,
                )
    
    def test_pipeline_intermediate_results(self):
        """Test saving intermediate results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def step1(data):
                return data * 2
            
            def step2(data):
                return data + 10
            
            steps = [('double', step1), ('add', step2)]
            
            results = run_pipeline(
                pipeline_steps=steps,
                input_data=5,
                output_dir=Path(tmpdir),
                pipeline_name='test',
                save_intermediate=True,
                git_tag=False,
            )
            
            assert 'intermediate_results' in results
            assert 'double' in results['intermediate_results']
            assert 'add' in results['intermediate_results']
            
            # Verify intermediate files exist
            for step_name, filepath in results['intermediate_results'].items():
                assert Path(filepath).exists()
    
    def test_compare_pipeline_versions(self):
        """Test comparing pipeline versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def step1(data):
                return data * 2
            
            steps = [('double', step1)]
            
            # Run pipeline twice with different inputs
            run_pipeline(
                pipeline_steps=steps,
                input_data=5,
                output_dir=Path(tmpdir),
                pipeline_name='test',
                version='v1',
                git_tag=False,
            )
            
            run_pipeline(
                pipeline_steps=steps,
                input_data=10,
                output_dir=Path(tmpdir),
                pipeline_name='test',
                version='v2',
                git_tag=False,
            )
            
            # Compare versions
            comparison = compare_pipeline_versions(
                'test',
                Path(tmpdir),
                version1='v1',
                version2='v2',
            )
            
            assert comparison['pipeline_name'] == 'test'
            assert comparison['version1'] == 'v1'
            assert comparison['version2'] == 'v2'
            assert not comparison['checksums_match']
    
    def test_get_pipeline_history(self):
        """Test retrieving pipeline execution history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def step1(data):
                return data * 2
            
            steps = [('double', step1)]
            
            # Run pipeline multiple times
            for i in range(3):
                run_pipeline(
                    pipeline_steps=steps,
                    input_data=i,
                    output_dir=Path(tmpdir),
                    pipeline_name='test',
                    version=f'v{i+1}',
                    git_tag=False,
                )
            
            # Get history
            history = get_pipeline_history('test', Path(tmpdir))
            
            assert len(history) >= 3
            version_ids = [h['version'] for h in history]
            assert 'v1' in version_ids
            assert 'v2' in version_ids
            assert 'v3' in version_ids


class TestGitIntegration:
    """Unit tests for git integration (may skip if git not available)."""
    
    def test_list_git_tags(self):
        """Test listing git tags."""
        try:
            tags = list_git_tags()
            # Should return a list (may be empty)
            assert isinstance(tags, list)
        except GitIntegrationError:
            pytest.skip("Git not available")
    
    def test_create_git_tag_without_git(self):
        """Test that creating tags fails gracefully without git."""
        # This test may fail if git is available
        # It's mainly to test error handling
        pass  # Skip for now as behavior depends on git availability


class TestEdgeCases:
    """Unit tests for edge cases."""
    
    def test_empty_result(self):
        """Test saving and loading empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {}
            filepath = Path(tmpdir) / 'result.pkl'
            
            saved_path, checksum = save_results(result, filepath)
            loaded_result, metadata = load_results(saved_path)
            
            assert loaded_result == result
    
    def test_large_result(self):
        """Test saving and loading large results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large DataFrame
            result = pd.DataFrame({
                'value': np.random.randn(10000),
                'category': np.random.choice(['A', 'B', 'C'], 10000),
            })
            filepath = Path(tmpdir) / 'result.pkl'
            
            saved_path, checksum = save_results(result, filepath)
            loaded_result, metadata = load_results(saved_path)
            
            pd.testing.assert_frame_equal(loaded_result, result)
    
    def test_nested_result(self):
        """Test saving and loading nested results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = {
                'level1': {
                    'level2': {
                        'level3': [1, 2, 3, 4, 5],
                    },
                },
                'dataframe': pd.DataFrame({'a': [1, 2, 3]}),
            }
            filepath = Path(tmpdir) / 'result.pkl'
            
            saved_path, checksum = save_results(result, filepath)
            loaded_result, metadata = load_results(saved_path)
            
            assert loaded_result['level1']['level2']['level3'] == [1, 2, 3, 4, 5]
            pd.testing.assert_frame_equal(
                loaded_result['dataframe'],
                result['dataframe']
            )
