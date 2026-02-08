"""Data acquisition module for fetching Wikipedia metrics from Wikimedia APIs."""

from .api_client import WikimediaAPIClient
from .data_validator import DataValidator
from .persistence import save_timeseries_data, load_timeseries_data

__all__ = [
    'WikimediaAPIClient',
    'DataValidator',
    'save_timeseries_data',
    'load_timeseries_data',
]
