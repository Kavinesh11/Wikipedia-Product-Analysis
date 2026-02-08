"""Visualization and Reporting.

Generates interactive dashboards and statistical evidence overlays.
"""

from wikipedia_health.visualization.plots import (
    plot_trend_with_confidence_bands,
    plot_campaign_effect,
    plot_forecast,
    plot_comparison,
    add_statistical_annotations,
)

from wikipedia_health.visualization.reports import (
    generate_summary_table,
    generate_finding_report,
    generate_evidence_report,
    export_report_to_file,
)

from wikipedia_health.visualization.dashboard import (
    create_interactive_trend_plot,
    create_interactive_campaign_plot,
    create_interactive_forecast_plot,
    create_interactive_comparison_plot,
    create_findings_dashboard,
)

__all__ = [
    'plot_trend_with_confidence_bands',
    'plot_campaign_effect',
    'plot_forecast',
    'plot_comparison',
    'add_statistical_annotations',
    'generate_summary_table',
    'generate_finding_report',
    'generate_evidence_report',
    'export_report_to_file',
    'create_interactive_trend_plot',
    'create_interactive_campaign_plot',
    'create_interactive_forecast_plot',
    'create_interactive_comparison_plot',
    'create_findings_dashboard',
]
