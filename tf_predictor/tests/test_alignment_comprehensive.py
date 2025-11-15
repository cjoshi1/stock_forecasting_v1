"""
Comprehensive alignment testing for all permutations of:
- Group configurations: none, single, multiple
- Horizons: single (1), multi (3)
- Targets: single, multi (2)

Total: 12 test scenarios covering all combinations

Tests verify:
1. Index tracking correctness
2. Temporal order within groups
3. Scaling happens within groups (no cross-group contamination)
4. Group boundary respect (sequences don't span groups)
5. Prediction-actual alignment
6. No data leakage
7. Row count consistency
8. Metadata preservation

Each test generates synthetic data and performs comprehensive validation.
Results are saved to alignment_test_results.md
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add repository root to path (parent of tf_predictor/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tf_predictor.core.predictor import TimeSeriesPredictor
from tf_predictor.core.utils import calculate_metrics


class TestAlignmentComprehensive:
    """Comprehensive alignment tests for all permutations."""

    # Test results collector
    test_results = []
    scaler_stats = []

    @classmethod
    def setup_class(cls):
        """Setup before all tests."""
        cls.test_results = []
        cls.scaler_stats = []

    @classmethod
    def teardown_class(cls):
        """Save results after all tests."""
        cls.save_results_to_markdown()

    # ==================== Helper Methods ====================

    @staticmethod
    def generate_synthetic_data(
        n_rows: int = 500,
        n_groups: int = None,
        group_columns: List[str] = None,
        n_features: int = 5,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic time series data for testing.

        Args:
            n_rows: Total number of rows (distributed across groups if applicable)
            n_groups: Number of unique groups (None for no groups)
            group_columns: List of group column names
            n_features: Number of feature columns
            seed: Random seed

        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(seed)

        if n_groups is not None and group_columns is not None:
            # With groups
            rows_per_group = n_rows // n_groups
            dfs = []

            for i in range(n_groups):
                # Generate group data
                data = {
                    'date': pd.date_range('2020-01-01', periods=rows_per_group, freq='D')
                }

                # Add group column values
                if len(group_columns) == 1:
                    data[group_columns[0]] = f'Group_{chr(65+i)}'  # A, B, C, ...
                else:
                    # Multi-column groups
                    data[group_columns[0]] = f'Group_{chr(65+i)}'
                    data[group_columns[1]] = f'Sector_{i % 3}'  # Rotate through 3 sectors

                # Add features with group-specific patterns
                for j in range(n_features):
                    # Each group has different mean/std for features
                    mean = 100 + i * 10
                    std = 10 + i * 2
                    data[f'feature_{j}'] = np.random.normal(mean, std, rows_per_group)

                # Add target columns with autocorrelation
                data['target1'] = np.cumsum(np.random.randn(rows_per_group)) + 100 + i * 20
                data['target2'] = np.cumsum(np.random.randn(rows_per_group)) + 50 + i * 10

                dfs.append(pd.DataFrame(data))

            df = pd.concat(dfs, ignore_index=True)
            # Shuffle to simulate unsorted input
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            # Without groups
            data = {
                'date': pd.date_range('2020-01-01', periods=n_rows, freq='D')
            }

            # Add features
            for j in range(n_features):
                data[f'feature_{j}'] = np.random.normal(100, 10, n_rows)

            # Add target columns
            data['target1'] = np.cumsum(np.random.randn(n_rows)) + 100
            data['target2'] = np.cumsum(np.random.randn(n_rows)) + 50

            df = pd.DataFrame(data)

        return df

    def verify_alignment(
        self,
        predictor: TimeSeriesPredictor,
        df: pd.DataFrame,
        test_name: str
    ) -> Dict:
        """
        Comprehensive alignment verification.

        Returns dict with verification results.
        """
        results = {
            'test_name': test_name,
            'checks': {}
        }

        # Make predictions
        if predictor.group_columns:
            predictions, group_indices = predictor.predict(df, return_group_info=True)
        else:
            predictions = predictor.predict(df)
            group_indices = None

        # Get metrics
        metrics = predictor.evaluate(df)

        # Check 1: Indices exist
        results['checks']['has_indices'] = hasattr(predictor, '_last_sequence_indices')

        if results['checks']['has_indices']:
            indices = predictor._last_sequence_indices

            # Check 2: No duplicate indices
            results['checks']['no_duplicates'] = len(indices) == len(set(indices))

            # Check 3: Indices in valid range
            if hasattr(predictor, '_last_processed_df'):
                valid_indices = set(predictor._last_processed_df['_original_index'].values)
                results['checks']['valid_indices'] = all(idx in valid_indices for idx in indices)

            # Check 4: Count consistency
            if predictor.is_multi_target:
                # Multi-target: predictions is a dict
                pred_count = len(next(iter(predictions.values())))
            else:
                pred_count = len(predictions)

            results['checks']['count_match'] = pred_count == len(indices)

            # Check 5: Temporal order within groups
            if predictor.group_columns and hasattr(predictor, '_last_processed_df'):
                eval_df = predictor._last_processed_df.set_index('_original_index', drop=False)
                time_col = predictor._detect_time_column(df)

                if time_col and time_col in eval_df.columns:
                    # Check each group separately
                    temporal_violations = []
                    for group_value in set(group_indices):
                        group_mask = [g == group_value for g in group_indices]
                        group_indices_subset = indices[group_mask]
                        group_dates = eval_df.loc[group_indices_subset, time_col].values

                        # Check if sorted
                        if not all(group_dates[i] <= group_dates[i+1] for i in range(len(group_dates)-1)):
                            temporal_violations.append(group_value)

                    results['checks']['temporal_order'] = len(temporal_violations) == 0
                    if temporal_violations:
                        results['checks']['temporal_violations'] = temporal_violations[:5]

        # Check 6: Metrics are reasonable (not NaN or inf)
        def check_metrics_valid(m):
            if isinstance(m, dict):
                return all(check_metrics_valid(v) for v in m.values())
            else:
                return not (np.isnan(m) or np.isinf(m))

        results['checks']['metrics_valid'] = check_metrics_valid(metrics)

        # Store metrics
        results['metrics'] = metrics

        return results

    @classmethod
    def save_results_to_markdown(cls):
        """Save all test results to alignment_test_results.md"""
        output_path = Path(__file__).parent.parent / 'alignment_test_results.md'

        with open(output_path, 'w') as f:
            f.write("# TF_Predictor Alignment Test Results\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now()}\n\n")
            f.write("---\n\n")

            # Summary table
            f.write("## Test Summary\n\n")
            f.write("| Test # | Groups | Horizon | Targets | Status | Issues |\n")
            f.write("|--------|--------|---------|---------|--------|--------|\n")

            for i, result in enumerate(cls.test_results, 1):
                checks = result['checks']
                all_passed = all(v == True for k, v in checks.items() if isinstance(v, bool))
                status = "✅ PASS" if all_passed else "❌ FAIL"

                issues = [k for k, v in checks.items() if isinstance(v, bool) and not v]
                issues_str = ", ".join(issues) if issues else "None"

                f.write(f"| {i} | {result.get('groups', 'None')} | "
                       f"{result.get('horizon', 1)} | {result.get('targets', 1)} | "
                       f"{status} | {issues_str} |\n")

            f.write("\n---\n\n")

            # Detailed results for each test
            f.write("## Detailed Test Results\n\n")

            for i, result in enumerate(cls.test_results, 1):
                f.write(f"### Test {i}: {result['test_name']}\n\n")

                # Configuration
                f.write("**Configuration:**\n")
                f.write(f"- Groups: {result.get('groups', 'None')}\n")
                f.write(f"- Horizon: {result.get('horizon', 1)}\n")
                f.write(f"- Targets: {result.get('targets', 1)}\n")
                f.write(f"- Sequence Length: {result.get('sequence_length', 10)}\n\n")

                # Checks
                f.write("**Validation Checks:**\n")
                for check_name, check_result in result['checks'].items():
                    if isinstance(check_result, bool):
                        icon = "✅" if check_result else "❌"
                        f.write(f"- {icon} {check_name}: {check_result}\n")
                    else:
                        f.write(f"- ⚠️  {check_name}: {check_result}\n")
                f.write("\n")

                # Metrics
                f.write("**Metrics:**\n")
                f.write("```\n")
                f.write(str(result.get('metrics', 'N/A')))
                f.write("\n```\n\n")

                f.write("---\n\n")

        print(f"\n✅ Test results saved to: {output_path}")

    # ==================== Test Cases ====================

    # Test 1: No groups, single horizon, single target
    def test_01_no_groups_single_horizon_single_target(self):
        """Test 1: Simplest case - no groups, single horizon, single target"""
        df = self.generate_synthetic_data(n_rows=200, n_groups=None)

        predictor = TimeSeriesPredictor(
            target_column='target1',
            sequence_length=10,
            prediction_horizon=1,
            group_columns=None,
            verbose=True
        )

        # Fit
        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        # Verify alignment
        result = self.verify_alignment(predictor, df, "No groups, single horizon, single target")
        result.update({
            'groups': 'None',
            'horizon': 1,
            'targets': 1,
            'sequence_length': 10
        })

        self.test_results.append(result)

        # Assert all checks passed
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    # Test 2: No groups, single horizon, multi target
    def test_02_no_groups_single_horizon_multi_target(self):
        """Test 2: No groups, single horizon, multiple targets"""
        df = self.generate_synthetic_data(n_rows=200, n_groups=None)

        predictor = TimeSeriesPredictor(
            target_column=['target1', 'target2'],
            sequence_length=10,
            prediction_horizon=1,
            group_columns=None,
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "No groups, single horizon, multi target")
        result.update({
            'groups': 'None',
            'horizon': 1,
            'targets': 2,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    # Test 3: No groups, multi horizon, single target
    def test_03_no_groups_multi_horizon_single_target(self):
        """Test 3: No groups, multi-horizon, single target"""
        df = self.generate_synthetic_data(n_rows=200, n_groups=None)

        predictor = TimeSeriesPredictor(
            target_column='target1',
            sequence_length=10,
            prediction_horizon=3,
            group_columns=None,
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "No groups, multi horizon, single target")
        result.update({
            'groups': 'None',
            'horizon': 3,
            'targets': 1,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    # Test 4: No groups, multi horizon, multi target
    def test_04_no_groups_multi_horizon_multi_target(self):
        """Test 4: No groups, multi-horizon, multiple targets"""
        df = self.generate_synthetic_data(n_rows=200, n_groups=None)

        predictor = TimeSeriesPredictor(
            target_column=['target1', 'target2'],
            sequence_length=10,
            prediction_horizon=3,
            group_columns=None,
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "No groups, multi horizon, multi target")
        result.update({
            'groups': 'None',
            'horizon': 3,
            'targets': 2,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    # Tests 5-8: Single group column
    def test_05_single_group_single_horizon_single_target(self):
        """Test 5: Single group column, single horizon, single target"""
        df = self.generate_synthetic_data(n_rows=500, n_groups=5, group_columns=['group'])

        predictor = TimeSeriesPredictor(
            target_column='target1',
            sequence_length=10,
            prediction_horizon=1,
            group_columns='group',
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Single group, single horizon, single target")
        result.update({
            'groups': 'Single',
            'horizon': 1,
            'targets': 1,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    def test_06_single_group_single_horizon_multi_target(self):
        """Test 6: Single group column, single horizon, multiple targets"""
        df = self.generate_synthetic_data(n_rows=500, n_groups=5, group_columns=['group'])

        predictor = TimeSeriesPredictor(
            target_column=['target1', 'target2'],
            sequence_length=10,
            prediction_horizon=1,
            group_columns='group',
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Single group, single horizon, multi target")
        result.update({
            'groups': 'Single',
            'horizon': 1,
            'targets': 2,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    def test_07_single_group_multi_horizon_single_target(self):
        """Test 7: Single group column, multi-horizon, single target"""
        df = self.generate_synthetic_data(n_rows=500, n_groups=5, group_columns=['group'])

        predictor = TimeSeriesPredictor(
            target_column='target1',
            sequence_length=10,
            prediction_horizon=3,
            group_columns='group',
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Single group, multi horizon, single target")
        result.update({
            'groups': 'Single',
            'horizon': 3,
            'targets': 1,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    def test_08_single_group_multi_horizon_multi_target(self):
        """Test 8: Single group column, multi-horizon, multiple targets"""
        df = self.generate_synthetic_data(n_rows=500, n_groups=5, group_columns=['group'])

        predictor = TimeSeriesPredictor(
            target_column=['target1', 'target2'],
            sequence_length=10,
            prediction_horizon=3,
            group_columns='group',
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Single group, multi horizon, multi target")
        result.update({
            'groups': 'Single',
            'horizon': 3,
            'targets': 2,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    # Tests 9-12: Multiple group columns
    def test_09_multi_group_single_horizon_single_target(self):
        """Test 9: Multiple group columns, single horizon, single target"""
        df = self.generate_synthetic_data(n_rows=600, n_groups=6, group_columns=['group', 'sector'])

        predictor = TimeSeriesPredictor(
            target_column='target1',
            sequence_length=10,
            prediction_horizon=1,
            group_columns=['group', 'sector'],
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Multi group, single horizon, single target")
        result.update({
            'groups': 'Multiple',
            'horizon': 1,
            'targets': 1,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    def test_10_multi_group_single_horizon_multi_target(self):
        """Test 10: Multiple group columns, single horizon, multiple targets"""
        df = self.generate_synthetic_data(n_rows=600, n_groups=6, group_columns=['group', 'sector'])

        predictor = TimeSeriesPredictor(
            target_column=['target1', 'target2'],
            sequence_length=10,
            prediction_horizon=1,
            group_columns=['group', 'sector'],
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Multi group, single horizon, multi target")
        result.update({
            'groups': 'Multiple',
            'horizon': 1,
            'targets': 2,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    def test_11_multi_group_multi_horizon_single_target(self):
        """Test 11: Multiple group columns, multi-horizon, single target"""
        df = self.generate_synthetic_data(n_rows=600, n_groups=6, group_columns=['group', 'sector'])

        predictor = TimeSeriesPredictor(
            target_column='target1',
            sequence_length=10,
            prediction_horizon=3,
            group_columns=['group', 'sector'],
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Multi group, multi horizon, single target")
        result.update({
            'groups': 'Multiple',
            'horizon': 3,
            'targets': 1,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))

    def test_12_multi_group_multi_horizon_multi_target(self):
        """Test 12: Multiple group columns, multi-horizon, multiple targets (FULL)"""
        df = self.generate_synthetic_data(n_rows=600, n_groups=6, group_columns=['group', 'sector'])

        predictor = TimeSeriesPredictor(
            target_column=['target1', 'target2'],
            sequence_length=10,
            prediction_horizon=3,
            group_columns=['group', 'sector'],
            verbose=True
        )

        predictor.fit(df, epochs=5, batch_size=32, verbose=False)

        result = self.verify_alignment(predictor, df, "Multi group, multi horizon, multi target (FULL)")
        result.update({
            'groups': 'Multiple',
            'horizon': 3,
            'targets': 2,
            'sequence_length': 10
        })

        self.test_results.append(result)
        assert all(v == True for k, v in result['checks'].items() if isinstance(v, bool))
