"""
Error analysis module.

Provides detailed analysis of model errors.
"""

from src.models.classical.analysis.error_analysis import (
    analyze_errors,
    analyze_false_negatives,
    analyze_false_positives,
    create_error_report
)
from src.models.classical.analysis.confusion_examples import (
    get_confusion_examples,
    plot_confusion_matrix_with_examples
)
from src.models.classical.analysis.edge_cases import (
    identify_edge_cases,
    create_edge_case_report
)

__all__ = [
    "analyze_errors",
    "analyze_false_negatives",
    "analyze_false_positives",
    "create_error_report",
    "get_confusion_examples",
    "plot_confusion_matrix_with_examples",
    "identify_edge_cases",
    "create_edge_case_report",
]
