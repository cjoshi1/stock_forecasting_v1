"""
Test to ensure tf_predictor has no dependencies on domain modules.
This keeps the module generic and reusable.
"""

import os
import re
from pathlib import Path

def test_no_domain_imports():
    """Ensure tf_predictor doesn't import from domain modules."""

    # Forbidden imports - these would couple tf_predictor to specific applications
    forbidden_patterns = [
        r'from daily_stock_forecasting',
        r'import daily_stock_forecasting',
        r'from intraday_forecasting',
        r'import intraday_forecasting',
        r'from.*stock_features',  # Domain-specific feature files
        r'from.*intraday_features',
        r'from.*market_data',
        r'from.*stock_charts',
        r'from.*intraday_charts',
    ]

    # Get tf_predictor directory (parent of tests directory)
    tf_predictor_dir = Path(__file__).parent.parent
    violations = []

    # Check all Python files in tf_predictor
    for py_file in tf_predictor_dir.rglob('*.py'):
        # Skip cache and test files
        if '__pycache__' in str(py_file) or py_file.name == 'test_no_domain_imports.py':
            continue

        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for forbidden patterns
        for line_num, line in enumerate(content.split('\n'), 1):
            # Skip comments and docstrings
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue

            for pattern in forbidden_patterns:
                if re.search(pattern, line):
                    relative_path = py_file.relative_to(tf_predictor_dir.parent)
                    violations.append(
                        f"{relative_path}:{line_num} - Found forbidden import: {line.strip()}"
                    )

    if violations:
        error_msg = "\n".join(violations)
        raise AssertionError(
            f"\n{'='*80}\n"
            f"ERROR: tf_predictor should not import from domain modules!\n"
            f"{'='*80}\n"
            f"tf_predictor is a GENERIC time series library and should have\n"
            f"NO dependencies on domain-specific modules like:\n"
            f"  - daily_stock_forecasting\n"
            f"  - intraday_forecasting\n"
            f"\n"
            f"Violations found:\n"
            f"{error_msg}\n"
            f"{'='*80}\n"
        )

if __name__ == '__main__':
    test_no_domain_imports()
    print("=" * 80)
    print("✓ BOUNDARY CHECK PASSED")
    print("=" * 80)
    print("tf_predictor has no dependencies on domain modules.")
    print("The module is properly decoupled and reusable!")
    print("=" * 80)
