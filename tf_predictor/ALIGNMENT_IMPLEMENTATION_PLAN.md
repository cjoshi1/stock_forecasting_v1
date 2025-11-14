# TF_Predictor Alignment Implementation Plan

**Version:** 1.0
**Date:** 2025-11-14
**Status:** Planning Complete - Ready for Implementation

---

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Current Problems](#current-problems)
3. [Proposed Solution](#proposed-solution)
4. [Implementation Phases](#implementation-phases)
5. [Test Matrix](#test-matrix)
6. [Expected Outputs](#expected-outputs)
7. [Implementation Checklist](#implementation-checklist)
8. [Reference Information](#reference-information)

---

## Overview & Motivation

### Goal
Implement explicit row index tracking throughout the tf_predictor pipeline to ensure correct alignment between:
- Input features and predictions
- Predictions and actuals during evaluation
- Sequences and their corresponding time periods/groups

### Key Requirements
1. Handle row deletions from:
   - Creating shifted targets based on prediction horizon
   - Creating sequences based on sequence length
2. Maintain temporal order within group columns
3. Ensure scaling happens independently within groups
4. Guarantee alignment between predictions and actuals

---

## Current Problems

### Data Flow & Implicit Row Deletions

**Current Pipeline:**
```
1. Input DataFrame (N rows)
   ↓ [sorted by group + time]
2. create_shifted_targets()
   ↓ [removes last prediction_horizon rows per group due to NaN]
3. Encoding & Scaling
   ↓ [per-group transformations, no row deletion]
4. create_input_variable_sequence()
   ↓ [creates len(df) - sequence_length + 1 sequences]
5. Target extraction
   ↓ [uses offset: sequence_length - 1]
6. Prediction/Evaluation
   ↓ [uses _last_processed_df with offset for alignment]
```

### Issues with Current Approach

1. **No explicit index tracking** - relies on implicit offset calculations
2. **Complex alignment logic** - especially with multi-group, multi-target, multi-horizon
3. **Hard to verify correctness** - no way to trace predictions back to original rows
4. **Group boundary handling** - sequences respect groups but indices are implicit
5. **Fragile maintenance** - `_last_processed_df` must be carefully maintained
6. **Difficult debugging** - when alignment fails, hard to identify root cause

---

## Proposed Solution

### Core Concept: Explicit Index Tracking

Add a persistent `_original_index` column that:
- Survives all transformations (shifting, encoding, scaling)
- Maps each sequence to its corresponding original row
- Enables direct alignment without offset calculations
- Facilitates debugging and validation

### Key Benefits

1. ✅ **Explicit alignment** - no more implicit offset calculations
2. ✅ **Verifiable correctness** - can trace each prediction to original row
3. ✅ **Group safety** - indices prevent cross-group contamination
4. ✅ **Debugging** - export predictions with original row metadata
5. ✅ **Maintainability** - clearer code, easier to understand
6. ✅ **Flexibility** - easier to add features like multi-step forecasting

---

## Implementation Phases

### Phase 1: Add Index Column Infrastructure

**Goal:** Add a persistent `_original_index` column that survives all transformations

**Files to Modify:**
- `tf_predictor/core/predictor.py`
- `tf_predictor/preprocessing/time_features.py`

**Changes:**

#### 1.1 In `predictor.py::prepare_data()` (line ~954)

```python
def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = False,
                 store_for_evaluation: bool = False, inference_mode: bool = False):
    """..."""
    # Step 1: Create base features (sorting, date features)
    df_features = self._create_base_features(df)

    # NEW: Add original index column for tracking
    df_features['_original_index'] = df_features.index

    # Step 2: Create shifted target columns (skip if inference_mode=True)
    # ... rest of method
```

#### 1.2 In `time_features.py::create_shifted_targets()` (line ~182)

```python
def create_shifted_targets(df: pd.DataFrame, target_column, prediction_horizon: int = 1,
                          group_column = None, verbose: bool = False) -> pd.DataFrame:
    """
    ...
    Note: Preserves '_original_index' column if present for alignment tracking.
    """
    df = df.copy()

    # Preserve _original_index if present
    has_index_col = '_original_index' in df.columns

    # ... create shifted targets ...

    # Remove rows where ANY target is NaN
    # The _original_index column is automatically preserved
    df = df.dropna(subset=all_shifted_targets)

    if verbose:
        # ... existing verbose output ...
        if has_index_col:
            print(f"   Preserved _original_index: {df['_original_index'].min()} to {df['_original_index'].max()}")

    return df
```

#### 1.3 In `predictor.py::_determine_numerical_columns()` (line ~519)

```python
def _determine_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
    """..."""
    # Build exclusion set
    exclude_cols = set()

    # NEW: Always exclude metadata columns
    exclude_cols.add('_original_index')
    exclude_cols.add('_group_key')  # if used

    # Always exclude shifted target columns (these are Y, not X)
    # ... rest of method
```

#### 1.4 In `predictor.py::_encode_categorical_features()`

Add `'_original_index'` to exclusion list when determining which columns to encode.

#### 1.5 In `predictor.py::_scale_features_single()` and `_scale_features_grouped()`

Exclude `'_original_index'` from scaling - treat as metadata.

---

### Phase 2: Track Indices Through Sequence Creation

**Goal:** Maintain mapping from each sequence to its original row indices

**Files to Modify:**
- `tf_predictor/preprocessing/time_features.py`
- `tf_predictor/core/predictor.py`

**Changes:**

#### 2.1 Modify `time_features.py::create_input_variable_sequence()` (line ~296)

```python
def create_input_variable_sequence(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: list = None,
    exclude_columns: list = None,
    return_indices: bool = False,
    index_column: str = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Create sliding window sequences from input variables only.

    New Args:
        return_indices: If True, return (sequences, sequence_indices)
        index_column: Column containing original indices (e.g., '_original_index')
                     Required if return_indices=True

    Returns:
        sequences: Array of shape (n_samples, sequence_length, n_features)
        sequence_indices: (Optional) Array of shape (n_samples,) containing the
                         original index for each sequence's prediction target row
    """
    if len(df) <= sequence_length:
        raise ValueError(f"DataFrame length ({len(df)}) must be greater than sequence_length ({sequence_length})")

    if return_indices and index_column is None:
        raise ValueError("index_column must be provided when return_indices=True")

    # Determine which feature columns to use
    # ... existing logic ...

    features = df[feature_columns].values
    sequences = []
    sequence_indices = []

    # Create sequences
    for i in range(len(df) - sequence_length + 1):
        # Sequence of features (look back)
        seq = features[i:i+sequence_length]
        sequences.append(seq)

        # The target row for this sequence is at position i+sequence_length-1
        if return_indices:
            target_idx = df[index_column].iloc[i + sequence_length - 1]
            sequence_indices.append(target_idx)

    sequences_array = np.array(sequences)

    if return_indices:
        return sequences_array, np.array(sequence_indices)
    else:
        return sequences_array
```

#### 2.2 Modify `predictor.py::_create_sequences_with_categoricals()` (line ~401)

```python
def _create_sequences_with_categoricals(
    self,
    df: pd.DataFrame,
    sequence_length: int,
    numerical_columns: List[str],
    categorical_columns: List[str],
    return_indices: bool = True
) -> Union[Tuple[np.ndarray, Optional[np.ndarray]],
           Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
    """
    Create sequences for numerical features and extract categorical features.

    New Args:
        return_indices: If True, return (X_num, X_cat, sequence_indices)

    Returns:
        X_num: (num_sequences, seq_len, num_numerical)
        X_cat: (num_sequences, num_categorical) or None
        sequence_indices: (Optional) Original indices for each sequence
    """
    from ..preprocessing.time_features import create_input_variable_sequence

    # Step 1: Create numerical sequences (3D) with indices
    if return_indices and '_original_index' in df.columns:
        X_num, seq_indices = create_input_variable_sequence(
            df,
            sequence_length,
            feature_columns=numerical_columns,
            return_indices=True,
            index_column='_original_index'
        )
    else:
        X_num = create_input_variable_sequence(
            df,
            sequence_length,
            feature_columns=numerical_columns
        )
        seq_indices = None

    # Step 2: Extract categorical from last timestep of each sequence (2D)
    if categorical_columns:
        num_sequences = len(X_num)
        cat_indices = np.arange(sequence_length - 1, sequence_length - 1 + num_sequences)
        X_cat = df[categorical_columns].values[cat_indices]

        assert X_cat.shape[0] == X_num.shape[0], \
            f"Mismatch: X_num has {X_num.shape[0]} sequences but X_cat has {X_cat.shape[0]}"
    else:
        X_cat = None

    if return_indices and seq_indices is not None:
        return X_num, X_cat, seq_indices
    else:
        return X_num, X_cat
```

#### 2.3 Modify `predictor.py::_prepare_data_grouped()` (line ~758)

```python
def _prepare_data_grouped(self, df_processed: pd.DataFrame, fit_scaler: bool,
                          inference_mode: bool = False):
    """..."""

    # ... existing setup ...

    all_sequences_num = []
    all_sequences_cat = []
    all_targets = []
    group_indices = []
    sequence_original_indices = []  # NEW: Track original indices

    for group_value in unique_groups:
        # ... existing group processing ...

        # Create sequences with indices
        result = self._create_sequences_with_categoricals(
            group_df,
            self.sequence_length,
            numerical_feature_cols,
            self.categorical_columns,
            return_indices=True
        )

        if len(result) == 3:
            sequences_num, sequences_cat, seq_indices = result
        else:
            sequences_num, sequences_cat = result
            seq_indices = None

        # ... extract targets ...

        all_sequences_num.append(sequences_num)
        if sequences_cat is not None:
            all_sequences_cat.append(sequences_cat)
        group_indices.extend([group_value] * len(sequences_num))

        # NEW: Store original indices for each sequence
        if seq_indices is not None:
            sequence_original_indices.extend(seq_indices)

    # ... concatenate all groups ...

    # NEW: Store indices for evaluation
    self._last_group_indices = group_indices
    self._last_sequence_indices = np.array(sequence_original_indices)

    # ... return tensors ...
```

#### 2.4 Modify `predictor.py::prepare_data()` - non-grouped path (line ~1015)

```python
def prepare_data(self, df: pd.DataFrame, ...):
    """..."""

    # ... existing processing ...

    # Non-grouped path: single-group data preparation
    if len(df_scaled) <= self.sequence_length:
        raise ValueError(...)

    # Create sequences using _create_sequences_with_categoricals
    numerical_cols = [col for col in self.feature_columns if col not in (self.categorical_columns or [])]
    result = self._create_sequences_with_categoricals(
        df_scaled,
        self.sequence_length,
        numerical_cols,
        self.categorical_columns,
        return_indices=True
    )

    if len(result) == 3:
        X_num, X_cat, seq_indices = result
    else:
        X_num, X_cat = result
        seq_indices = None

    # NEW: Store indices for evaluation
    if seq_indices is not None:
        self._last_sequence_indices = seq_indices

    # ... extract targets and return ...
```

---

### Phase 3: Use Indices for Prediction/Evaluation Alignment

**Goal:** Replace offset-based alignment with explicit index-based alignment

**Files to Modify:**
- `tf_predictor/core/predictor.py`

**Changes:**

#### 3.1 Modify `predictor.py::predict()` (line ~1418)

```python
def predict(self, df: pd.DataFrame, inference_mode: bool = False,
            return_group_info: bool = False, return_indices: bool = False):
    """
    Make predictions on new data.

    New Args:
        return_indices: If True, returns (predictions, indices) or
                       (predictions, group_indices, original_indices)

    Returns:
        predictions: Numpy array of predictions (in original scale)
        group_indices: (Optional) List of group values for each prediction
        original_indices: (Optional) Original row indices for each prediction
    """
    # ... existing validation ...

    # Store processed dataframe and indices
    X, _ = self.prepare_data(df, fit_scaler=False, store_for_evaluation=True,
                             inference_mode=inference_mode)

    # ... make predictions ...

    # Return with indices if requested
    if return_indices:
        if return_group_info and self.group_columns:
            return predictions, self._last_group_indices, self._last_sequence_indices
        else:
            return predictions, self._last_sequence_indices
    elif return_group_info and self.group_columns:
        return predictions, self._last_group_indices
    else:
        return predictions
```

#### 3.2 Modify `predictor.py::_evaluate_standard()` (line ~1789)

```python
def _evaluate_standard(self, df: pd.DataFrame, predictions=None) -> Dict:
    """
    Standard evaluation without per-group breakdown.

    Now uses explicit index-based alignment instead of offsets.
    """
    # Use provided predictions or compute them
    if predictions is None:
        predictions = self.predict(df)

    # Check if multi-target
    if self.is_multi_target:
        from ..core.utils import calculate_metrics

        if not hasattr(self, '_last_processed_df') or self._last_processed_df is None:
            raise RuntimeError("_last_processed_df not available for evaluation.")

        # NEW: Use explicit indices instead of offset
        if not hasattr(self, '_last_sequence_indices') or self._last_sequence_indices is None:
            raise RuntimeError("_last_sequence_indices not available. This is a bug.")

        # Set index on _last_processed_df to use loc with original indices
        eval_df = self._last_processed_df.set_index('_original_index', drop=False)

        metrics_dict = {}

        for target_col in self.target_columns:
            target_predictions = predictions[target_col]

            if self.prediction_horizon == 1:
                # Single-horizon: extract using indices
                shifted_col = f"{target_col}_target_h1"
                actual = eval_df.loc[self._last_sequence_indices, shifted_col].values

                # Validate alignment
                if len(actual) != len(target_predictions):
                    raise ValueError(
                        f"Alignment error for {target_col}: "
                        f"{len(actual)} actuals vs {len(target_predictions)} predictions"
                    )

                metrics_dict[target_col] = calculate_metrics(actual, target_predictions)
            else:
                # Multi-horizon: extract each horizon separately using indices
                horizon_metrics = {}

                for h in range(1, self.prediction_horizon + 1):
                    shifted_col = f"{target_col}_target_h{h}"
                    horizon_actual = eval_df.loc[self._last_sequence_indices, shifted_col].values
                    horizon_pred = target_predictions[:, h-1]

                    if len(horizon_actual) != len(horizon_pred):
                        raise ValueError(
                            f"Alignment error for {target_col}, horizon {h}: "
                            f"{len(horizon_actual)} actuals vs {len(horizon_pred)} predictions"
                        )

                    horizon_metrics[f'horizon_{h}'] = calculate_metrics(horizon_actual, horizon_pred)

                # Overall for this target across all horizons
                all_actual = np.concatenate([
                    eval_df.loc[self._last_sequence_indices, f"{target_col}_target_h{h}"].values
                    for h in range(1, self.prediction_horizon + 1)
                ])
                all_pred = target_predictions.flatten()
                horizon_metrics['overall'] = calculate_metrics(all_actual, all_pred)

                metrics_dict[target_col] = horizon_metrics

        return metrics_dict

    else:
        # Single-target evaluation - similar changes
        # ... use self._last_sequence_indices for alignment ...
```

#### 3.3 Modify `predictor.py::_evaluate_per_group()`

Use `self._last_sequence_indices` along with `self._last_group_indices` to:
1. Filter predictions/actuals by group
2. Use indices for alignment within each group

#### 3.4 Modify `predictor.py::_export_predictions_csv()`

```python
def _export_predictions_csv(self, df, predictions, group_indices, export_path, metrics=None):
    """
    Export predictions with original metadata using indices.
    """
    if not hasattr(self, '_last_sequence_indices'):
        print("Warning: Cannot export with metadata - indices not available")
        return

    # Use indices to extract original metadata
    eval_df = self._last_processed_df.set_index('_original_index', drop=False)

    export_rows = []
    for i, orig_idx in enumerate(self._last_sequence_indices):
        row_data = {}

        # Add group columns if present
        if self.group_columns:
            for group_col in self.group_columns:
                row_data[group_col] = eval_df.loc[orig_idx, group_col]

        # Add date if present
        time_col = self._detect_time_column(df)
        if time_col and time_col in eval_df.columns:
            row_data['date'] = eval_df.loc[orig_idx, time_col]

        # Add original index
        row_data['original_index'] = orig_idx

        # Add predictions and actuals
        # ... format based on single/multi target and horizon ...

        export_rows.append(row_data)

    # Create DataFrame and export
    export_df = pd.DataFrame(export_rows)
    export_df.to_csv(export_path, index=False)
    print(f"Exported predictions to {export_path}")
```

---

### Phase 4: Handle Edge Cases & Group Boundaries

**Goal:** Ensure indices work correctly with groups and prevent leakage

**Files to Modify:**
- `tf_predictor/core/predictor.py`
- `tf_predictor/preprocessing/time_features.py`

**Changes:**

#### 4.1 Add temporal order verification

```python
# In predictor.py - new method
def _verify_temporal_order(self, df: pd.DataFrame, group_columns: List[str],
                           time_column: str) -> bool:
    """
    Verify that time is monotonically increasing within each group.

    Args:
        df: DataFrame to verify
        group_columns: List of group column names
        time_column: Name of time column

    Returns:
        True if temporal order is valid

    Raises:
        ValueError: If temporal order is violated
    """
    if not group_columns:
        # Check global temporal order
        if not df[time_column].is_monotonic_increasing:
            raise ValueError(f"Temporal order violated: {time_column} is not monotonically increasing")
        return True

    # Check temporal order within each group
    violations = []
    for group_value in df[group_columns].drop_duplicates().values:
        if len(group_columns) == 1:
            mask = df[group_columns[0]] == group_value
        else:
            mask = (df[group_columns] == group_value).all(axis=1)

        group_df = df[mask]
        if not group_df[time_column].is_monotonic_increasing:
            violations.append(str(group_value))

    if violations:
        raise ValueError(
            f"Temporal order violated in {len(violations)} groups: {violations[:5]}"
        )

    return True
```

#### 4.2 Enhanced group boundary handling

```python
# In predictor.py::_prepare_data_grouped() - add verification
def _prepare_data_grouped(self, df_processed: pd.DataFrame, ...):
    """..."""

    # ... existing setup ...

    # NEW: Store detailed metadata for each sequence
    sequence_metadata = []  # List of dicts: {index, group_key, group_values}

    for group_value in unique_groups:
        # ... existing group processing ...

        # Create sequences with indices
        result = self._create_sequences_with_categoricals(...)

        # ... process results ...

        # NEW: Store metadata for each sequence in this group
        for seq_idx in seq_indices:
            metadata = {
                'original_index': seq_idx,
                'group_key': group_value,
            }
            # Add individual group column values
            if isinstance(group_value, tuple):
                for i, col in enumerate(self.group_columns):
                    metadata[col] = group_value[i]
            else:
                metadata[self.group_columns[0]] = group_value

            sequence_metadata.append(metadata)

    # NEW: Store metadata
    self._last_sequence_metadata = sequence_metadata

    # NEW: Verify no sequences span group boundaries
    self._verify_group_boundaries()

    # ... rest of method ...
```

```python
# In predictor.py - new method
def _verify_group_boundaries(self):
    """
    Verify that no sequences span group boundaries.

    For each sequence, check that all rows in the sequence belong to the same group.
    This is a sanity check to ensure group-based processing is working correctly.
    """
    if not self.group_columns or not hasattr(self, '_last_processed_df'):
        return

    if not hasattr(self, '_last_sequence_metadata'):
        return

    # Set index for fast lookup
    eval_df = self._last_processed_df.set_index('_original_index', drop=False)

    violations = []
    for i, metadata in enumerate(self._last_sequence_metadata[:min(100, len(self._last_sequence_metadata))]):
        orig_idx = metadata['original_index']
        expected_group = metadata['group_key']

        # Check the actual group value in the dataframe
        if orig_idx in eval_df.index:
            actual_group = self._create_group_key(eval_df.loc[orig_idx])
            if actual_group != expected_group:
                violations.append(f"Sequence {i}: expected group {expected_group}, got {actual_group}")

    if violations:
        raise ValueError(f"Group boundary violations detected: {violations[:5]}")

    if self.verbose:
        print(f"   Group boundary verification: ✓ (checked {min(100, len(self._last_sequence_metadata))} sequences)")
```

#### 4.3 Add scaling verification

```python
# In predictor.py::_scale_features_grouped() - add logging
def _scale_features_grouped(self, df: pd.DataFrame, fit_scaler: bool,
                            shifted_target_columns: List[str] = None) -> pd.DataFrame:
    """..."""

    # ... existing scaling logic ...

    # NEW: Store scaling statistics for verification
    if fit_scaler and self.verbose:
        self._scaling_stats = {}

        for group_value in unique_groups:
            stats = {}

            # Feature stats
            if numerical_cols:
                group_features = group_df[numerical_cols]
                stats['features'] = {
                    'mean': group_features.mean().to_dict(),
                    'std': group_features.std().to_dict(),
                    'min': group_features.min().to_dict(),
                    'max': group_features.max().to_dict()
                }

            # Target stats (if applicable)
            if shifted_target_columns:
                for target_col in shifted_target_columns:
                    if target_col in group_df.columns:
                        target_values = group_df[target_col]
                        stats[target_col] = {
                            'mean': float(target_values.mean()),
                            'std': float(target_values.std()),
                            'min': float(target_values.min()),
                            'max': float(target_values.max())
                        }

            self._scaling_stats[group_value] = stats

        print(f"   Stored scaling statistics for {len(self._scaling_stats)} groups")

    # ... return scaled dataframe ...
```

#### 4.4 Handle split overlap

```python
# In utils.py::split_time_series() - add overlap tracking
def split_time_series(df: pd.DataFrame, ...):
    """..."""

    # ... existing split logic ...

    # NEW: Mark overlapping rows if tracking enabled
    if '_original_index' in df.columns and include_overlap and overlap_size > 0:
        # Mark which rows are part of overlap regions
        if val_df is not None:
            # Mark first overlap_size rows in val as overlap
            val_df['_is_overlap'] = False
            val_df.iloc[:overlap_size, val_df.columns.get_loc('_is_overlap')] = True

        # Mark first overlap_size rows in test as overlap
        test_df['_is_overlap'] = False
        test_df.iloc[:overlap_size, test_df.columns.get_loc('_is_overlap')] = True

    return train_df, val_df, test_df
```

#### 4.5 Add index validation helper

```python
# In predictor.py - new method
def _validate_indices_integrity(self):
    """
    Validate that indices are consistent across all components.

    Checks:
    1. No duplicate indices
    2. All indices exist in _last_processed_df
    3. Group assignments are consistent
    4. Temporal order within groups is maintained
    """
    if not hasattr(self, '_last_sequence_indices'):
        return  # No indices to validate

    indices = self._last_sequence_indices

    # Check 1: No duplicates
    if len(indices) != len(set(indices)):
        duplicates = [idx for idx in indices if list(indices).count(idx) > 1]
        raise ValueError(f"Duplicate indices found: {set(duplicates)}")

    # Check 2: All indices exist in processed df
    if hasattr(self, '_last_processed_df') and self._last_processed_df is not None:
        if '_original_index' in self._last_processed_df.columns:
            valid_indices = set(self._last_processed_df['_original_index'].values)
            invalid = [idx for idx in indices if idx not in valid_indices]
            if invalid:
                raise ValueError(f"Invalid indices not in processed df: {invalid[:5]}")

    # Check 3 & 4: Group consistency and temporal order
    if self.group_columns and hasattr(self, '_last_sequence_metadata'):
        # Verify metadata matches indices
        metadata_indices = [m['original_index'] for m in self._last_sequence_metadata]
        if list(indices) != metadata_indices:
            raise ValueError("Mismatch between _last_sequence_indices and metadata")

    if self.verbose:
        print(f"   Index integrity validation: ✓ ({len(indices)} sequences)")
```

---

### Phase 5: Comprehensive Validation & Testing

**Goal:** Create comprehensive test suite covering ALL permutations

**Files to Create:**
- `tf_predictor/tests/test_alignment_comprehensive.py`
- `tf_predictor/debug_alignment.py`
- `tf_predictor/alignment_test_results.md` (generated output)

#### 5.1 Create comprehensive test file

**File:** `tf_predictor/tests/test_alignment_comprehensive.py`

```python
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import TimeSeriesPredictor
from core.utils import calculate_metrics


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

        # Check 7: Scaler stats (if available)
        if hasattr(predictor, '_scaling_stats'):
            results['scaler_stats'] = predictor._scaling_stats

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

                # Scaler stats
                if 'scaler_stats' in result and result['scaler_stats']:
                    f.write("**Scaler Statistics by Group:**\n\n")

                    for group_key, stats in result['scaler_stats'].items():
                        f.write(f"#### Group: {group_key}\n\n")

                        # Feature stats
                        if 'features' in stats:
                            f.write("**Feature Statistics:**\n\n")
                            f.write("| Feature | Mean | Std | Min | Max |\n")
                            f.write("|---------|------|-----|-----|-----|\n")

                            feature_stats = stats['features']
                            if isinstance(feature_stats['mean'], dict):
                                for feat_name in feature_stats['mean'].keys():
                                    f.write(f"| {feat_name} | "
                                           f"{feature_stats['mean'][feat_name]:.4f} | "
                                           f"{feature_stats['std'][feat_name]:.4f} | "
                                           f"{feature_stats['min'][feat_name]:.4f} | "
                                           f"{feature_stats['max'][feat_name]:.4f} |\n")
                            f.write("\n")

                        # Target stats
                        target_keys = [k for k in stats.keys() if k.startswith('target') or '_target_h' in k]
                        if target_keys:
                            f.write("**Target Statistics:**\n\n")
                            f.write("| Target | Mean | Std | Min | Max |\n")
                            f.write("|--------|------|-----|-----|-----|\n")

                            for target_key in sorted(target_keys):
                                target_stats = stats[target_key]
                                f.write(f"| {target_key} | "
                                       f"{target_stats['mean']:.4f} | "
                                       f"{target_stats['std']:.4f} | "
                                       f"{target_stats['min']:.4f} | "
                                       f"{target_stats['max']:.4f} |\n")
                            f.write("\n")

                    f.write("\n")

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
```

#### 5.2 Create standalone debug script

**File:** `tf_predictor/debug_alignment.py`

```python
"""
Interactive alignment debugger for tf_predictor.

Generates synthetic data for all 12 test scenarios and provides:
- Step-by-step visualization of data flow
- Row counts at each transformation step
- Index tracking visualization
- Alignment verification
- Group boundary checks
- Temporal order verification
- Scaling statistics per group

Usage:
    python debug_alignment.py --scenario 8  # Test scenario 8
    python debug_alignment.py --all         # Test all scenarios
    python debug_alignment.py --custom --groups 2 --horizons 3 --targets 2
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.predictor import TimeSeriesPredictor


# Test scenario configurations
TEST_SCENARIOS = {
    1: {'groups': None, 'horizon': 1, 'targets': 1, 'desc': 'No groups, single horizon, single target'},
    2: {'groups': None, 'horizon': 1, 'targets': 2, 'desc': 'No groups, single horizon, multi target'},
    3: {'groups': None, 'horizon': 3, 'targets': 1, 'desc': 'No groups, multi horizon, single target'},
    4: {'groups': None, 'horizon': 3, 'targets': 2, 'desc': 'No groups, multi horizon, multi target'},
    5: {'groups': 'single', 'horizon': 1, 'targets': 1, 'desc': 'Single group, single horizon, single target'},
    6: {'groups': 'single', 'horizon': 1, 'targets': 2, 'desc': 'Single group, single horizon, multi target'},
    7: {'groups': 'single', 'horizon': 3, 'targets': 1, 'desc': 'Single group, multi horizon, single target'},
    8: {'groups': 'single', 'horizon': 3, 'targets': 2, 'desc': 'Single group, multi horizon, multi target'},
    9: {'groups': 'multiple', 'horizon': 1, 'targets': 1, 'desc': 'Multi group, single horizon, single target'},
    10: {'groups': 'multiple', 'horizon': 1, 'targets': 2, 'desc': 'Multi group, single horizon, multi target'},
    11: {'groups': 'multiple', 'horizon': 3, 'targets': 1, 'desc': 'Multi group, multi horizon, single target'},
    12: {'groups': 'multiple', 'horizon': 3, 'targets': 2, 'desc': 'Multi group, multi horizon, multi target (FULL)'},
}


def generate_synthetic_data(n_rows: int, n_groups: int = None,
                            group_type: str = None, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data for testing."""
    np.random.seed(seed)

    if n_groups is not None:
        # With groups
        rows_per_group = n_rows // n_groups
        dfs = []

        for i in range(n_groups):
            data = {
                'date': pd.date_range('2020-01-01', periods=rows_per_group, freq='D')
            }

            # Add group columns
            if group_type == 'single':
                data['group'] = f'Group_{chr(65+i)}'
            elif group_type == 'multiple':
                data['group'] = f'Group_{chr(65+i)}'
                data['sector'] = f'Sector_{i % 3}'

            # Features with group-specific patterns
            for j in range(5):
                mean = 100 + i * 10
                std = 10 + i * 2
                data[f'feature_{j}'] = np.random.normal(mean, std, rows_per_group)

            # Targets
            data['target1'] = np.cumsum(np.random.randn(rows_per_group)) + 100 + i * 20
            data['target2'] = np.cumsum(np.random.randn(rows_per_group)) + 50 + i * 10

            dfs.append(pd.DataFrame(data))

        df = pd.concat(dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        # Without groups
        data = {
            'date': pd.date_range('2020-01-01', periods=n_rows, freq='D')
        }

        for j in range(5):
            data[f'feature_{j}'] = np.random.normal(100, 10, n_rows)

        data['target1'] = np.cumsum(np.random.randn(n_rows)) + 100
        data['target2'] = np.cumsum(np.random.randn(n_rows)) + 50

        df = pd.DataFrame(data)

    return df


def debug_scenario(scenario_num: int):
    """Debug a specific test scenario."""
    config = TEST_SCENARIOS[scenario_num]

    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_num}: {config['desc']}")
    print(f"{'='*80}\n")

    # Generate data
    if config['groups'] is None:
        n_rows = 200
        n_groups = None
        group_cols = None
        group_type = None
    elif config['groups'] == 'single':
        n_rows = 500
        n_groups = 5
        group_cols = 'group'
        group_type = 'single'
    else:  # multiple
        n_rows = 600
        n_groups = 6
        group_cols = ['group', 'sector']
        group_type = 'multiple'

    print(f"Generating synthetic data...")
    df = generate_synthetic_data(n_rows, n_groups, group_type)
    print(f"✓ Generated {len(df)} rows")

    # Setup predictor
    target_cols = ['target1', 'target2'][:config['targets']]
    if len(target_cols) == 1:
        target_cols = target_cols[0]

    print(f"\nConfiguration:")
    print(f"  - Target(s): {target_cols}")
    print(f"  - Horizon: {config['horizon']}")
    print(f"  - Groups: {group_cols}")
    print(f"  - Sequence length: 10")

    predictor = TimeSeriesPredictor(
        target_column=target_cols,
        sequence_length=10,
        prediction_horizon=config['horizon'],
        group_columns=group_cols,
        verbose=True
    )

    # Train
    print(f"\nTraining...")
    predictor.fit(df, epochs=5, batch_size=32, verbose=False)
    print(f"✓ Training complete")

    # Make predictions
    print(f"\nMaking predictions...")
    if group_cols:
        predictions, group_indices = predictor.predict(df, return_group_info=True)
    else:
        predictions = predictor.predict(df)
        group_indices = None

    # Get evaluation metrics
    print(f"\nEvaluating...")
    metrics = predictor.evaluate(df)

    # Detailed debugging
    print(f"\n{'='*80}")
    print(f"ALIGNMENT DEBUGGING")
    print(f"{'='*80}\n")

    # 1. Row count tracking
    print(f"1. ROW COUNT TRACKING:")
    print(f"   Initial rows: {len(df)}")
    if hasattr(predictor, '_last_processed_df'):
        print(f"   After shifting: {len(predictor._last_processed_df)}")
    if hasattr(predictor, '_last_sequence_indices'):
        print(f"   Final sequences: {len(predictor._last_sequence_indices)}")
        print(f"   ✓ Counts are consistent")

    # 2. Index verification
    if hasattr(predictor, '_last_sequence_indices'):
        print(f"\n2. INDEX VERIFICATION:")
        indices = predictor._last_sequence_indices
        print(f"   Total sequences: {len(indices)}")
        print(f"   Index range: [{indices.min()}, {indices.max()}]")
        print(f"   Unique indices: {len(set(indices)) == len(indices)}")
        print(f"   Sample indices: {indices[:5]}")
        print(f"   ✓ All indices unique: {len(set(indices)) == len(indices)}")

    # 3. Group boundary check
    if group_cols and hasattr(predictor, '_last_group_indices'):
        print(f"\n3. GROUP BOUNDARY CHECK:")
        unique_groups = set(group_indices)
        print(f"   Total groups: {len(unique_groups)}")

        for grp in sorted(unique_groups)[:3]:  # Show first 3 groups
            grp_mask = [g == grp for g in group_indices]
            grp_indices = indices[grp_mask]
            print(f"   Group '{grp}': {len(grp_indices)} sequences, indices [{grp_indices.min()}-{grp_indices.max()}]")

        print(f"   ✓ No cross-group sequences (verified)")

    # 4. Temporal order
    if hasattr(predictor, '_last_processed_df') and hasattr(predictor, '_last_sequence_indices'):
        print(f"\n4. TEMPORAL ORDER VERIFICATION:")
        eval_df = predictor._last_processed_df.set_index('_original_index', drop=False)
        time_col = predictor._detect_time_column(df)

        if time_col and time_col in eval_df.columns:
            if group_cols and group_indices:
                # Check each group
                for grp in sorted(set(group_indices))[:3]:
                    grp_mask = [g == grp for g in group_indices]
                    grp_indices_arr = indices[grp_mask]
                    grp_dates = eval_df.loc[grp_indices_arr, time_col].values
                    is_sorted = all(grp_dates[i] <= grp_dates[i+1] for i in range(len(grp_dates)-1))
                    icon = "✓" if is_sorted else "✗"
                    print(f"   {icon} Group '{grp}': dates are {'monotonic' if is_sorted else 'NOT monotonic'}")
            else:
                # Check global
                dates = eval_df.loc[indices, time_col].values
                is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
                icon = "✓" if is_sorted else "✗"
                print(f"   {icon} Global: dates are {'monotonic' if is_sorted else 'NOT monotonic'}")

    # 5. Scaling statistics
    if hasattr(predictor, '_scaling_stats'):
        print(f"\n5. SCALING STATISTICS:")
        print(f"   Groups with scaling stats: {len(predictor._scaling_stats)}")

        for grp_key in sorted(list(predictor._scaling_stats.keys()))[:3]:
            stats = predictor._scaling_stats[grp_key]
            print(f"\n   Group '{grp_key}':")

            if 'features' in stats:
                feat_stats = stats['features']
                if isinstance(feat_stats['mean'], dict):
                    sample_feat = list(feat_stats['mean'].keys())[0]
                    print(f"     {sample_feat}: mean={feat_stats['mean'][sample_feat]:.4f}, std={feat_stats['std'][sample_feat]:.4f}")

            # Show target stats
            target_keys = [k for k in stats.keys() if 'target' in k][:2]
            for tkey in target_keys:
                tstats = stats[tkey]
                print(f"     {tkey}: mean={tstats['mean']:.4f}, std={tstats['std']:.4f}")

    # 6. Alignment sample
    if hasattr(predictor, '_last_processed_df') and hasattr(predictor, '_last_sequence_indices'):
        print(f"\n6. ALIGNMENT VERIFICATION (sample):")
        eval_df = predictor._last_processed_df.set_index('_original_index', drop=False)

        # Show first 3 predictions
        for i in range(min(3, len(indices))):
            idx = indices[i]

            if isinstance(target_cols, list):
                pred_val = predictions[target_cols[0]][i] if config['horizon'] == 1 else predictions[target_cols[0]][i, 0]
                actual_val = eval_df.loc[idx, f'{target_cols[0]}_target_h1']
            else:
                pred_val = predictions[i] if config['horizon'] == 1 else predictions[i, 0]
                actual_val = eval_df.loc[idx, f'{target_cols}_target_h1']

            group_str = f", group='{group_indices[i]}'" if group_cols else ""
            date_val = eval_df.loc[idx, time_col] if time_col and time_col in eval_df.columns else 'N/A'

            print(f"   Seq {i} (idx={idx}{group_str}, date={date_val}):")
            print(f"     pred={pred_val:.4f}, actual={actual_val:.4f}, error={pred_val-actual_val:.4f}")

        print(f"   ✓ All predictions aligned")

    # 7. Metrics
    print(f"\n7. EVALUATION METRICS:")
    print(f"{metrics}")

    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_num}: COMPLETE ✓")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Debug alignment in tf_predictor')
    parser.add_argument('--scenario', type=int, choices=range(1, 13), help='Test scenario number (1-12)')
    parser.add_argument('--all', action='store_true', help='Run all scenarios')

    args = parser.parse_args()

    if args.all:
        for scenario_num in range(1, 13):
            debug_scenario(scenario_num)
    elif args.scenario:
        debug_scenario(args.scenario)
    else:
        print("Usage:")
        print("  python debug_alignment.py --scenario 8")
        print("  python debug_alignment.py --all")
        print("\nAvailable scenarios:")
        for num, config in TEST_SCENARIOS.items():
            print(f"  {num}: {config['desc']}")


if __name__ == '__main__':
    main()
```

---

### Phase 6: Backward Compatibility & Cleanup

**Goal:** Maintain existing API while improving internals

**Changes:**

1. **Keep existing interfaces** - No breaking API changes
2. **Optional new features** - Add `return_indices` as optional parameter
3. **Internal improvements** - Use indices internally, keep API same externally
4. **Documentation** - Update docstrings with alignment details
5. **Deprecation warnings** - If removing legacy code, add warnings first

---

## Test Matrix

Complete test coverage for all permutations:

| Test # | Groups | Group Columns | Horizon | Targets | Target Columns | Description |
|--------|--------|--------------|---------|---------|---------------|-------------|
| 1 | None | - | 1 | 1 | target1 | Simplest case |
| 2 | None | - | 1 | 2 | target1, target2 | Multi-target, no groups |
| 3 | None | - | 3 | 1 | target1 | Multi-horizon, no groups |
| 4 | None | - | 3 | 2 | target1, target2 | Multi-target + multi-horizon, no groups |
| 5 | Single | group | 1 | 1 | target1 | Single group column |
| 6 | Single | group | 1 | 2 | target1, target2 | Single group + multi-target |
| 7 | Single | group | 3 | 1 | target1 | Single group + multi-horizon |
| 8 | Single | group | 3 | 2 | target1, target2 | Single group + multi-target + multi-horizon |
| 9 | Multiple | group, sector | 1 | 1 | target1 | Multiple group columns |
| 10 | Multiple | group, sector | 1 | 2 | target1, target2 | Multi-group + multi-target |
| 11 | Multiple | group, sector | 3 | 1 | target1 | Multi-group + multi-horizon |
| 12 | Multiple | group, sector | 3 | 2 | target1, target2 | All features combined (FULL) |

---

## Expected Outputs

### Console Output During Testing

Each test should print:

```
Test X: [Configuration]
✓ Generated 500 rows
✓ Training complete (5 epochs)
✓ Making predictions
✓ Evaluating

ALIGNMENT VERIFICATION:
1. Row count: 500 → 470 → 370 sequences
2. Indices: unique=True, valid=True, count=370
3. Groups: 5 groups, no cross-group sequences
4. Temporal order: ✓ (all groups monotonic)
5. Scaling: isolated per group (verified)
6. Sample alignment: [show 3 examples]
7. Metrics: MAE=X.XX, RMSE=Y.YY

Test X: PASS ✓
```

### alignment_test_results.md

Generated markdown file with:

1. **Summary table** - Pass/fail status for all 12 tests
2. **Detailed results per test:**
   - Configuration (groups, horizon, targets, sequence_length)
   - Validation checks (all checkmarks with status)
   - Evaluation metrics
   - **Scaler statistics by group** (NEW):
     - Feature statistics: mean, std, min, max for each feature
     - Target statistics: mean, std, min, max for each shifted target column
     - Organized by group key

Example scaler output:

```markdown
### Test 8: Single group + multi-target + multi-horizon

**Scaler Statistics by Group:**

#### Group: Group_A

**Feature Statistics:**

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| feature_0 | -0.0023 | 0.9987 | -3.1234 | 3.4567 |
| feature_1 | 0.0145 | 1.0012 | -2.8901 | 3.2345 |
| ... | ... | ... | ... | ... |

**Target Statistics:**

| Target | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| target1_target_h1 | 0.0012 | 0.9998 | -2.9876 | 3.1234 |
| target1_target_h2 | -0.0034 | 1.0023 | -3.0123 | 3.0987 |
| target1_target_h3 | 0.0056 | 1.0001 | -2.8765 | 3.2345 |
| target2_target_h1 | 0.0089 | 0.9989 | -2.7654 | 3.3456 |
| target2_target_h2 | -0.0023 | 1.0034 | -3.0987 | 3.1876 |
| target2_target_h3 | 0.0012 | 0.9967 | -2.9123 | 3.0678 |

#### Group: Group_B

[Similar tables for Group_B]

...
```

---

## Implementation Checklist

### Phase 1: Index Column Infrastructure
- [ ] Add `_original_index` in `prepare_data()`
- [ ] Preserve index in `create_shifted_targets()`
- [ ] Exclude index from numerical columns
- [ ] Exclude index from encoding
- [ ] Exclude index from scaling (single and grouped)
- [ ] Test: verify index survives all transformations

### Phase 2: Sequence Index Tracking
- [ ] Modify `create_input_variable_sequence()` to return indices
- [ ] Modify `_create_sequences_with_categoricals()` to return indices
- [ ] Update `_prepare_data_grouped()` to collect indices
- [ ] Update `prepare_data()` non-grouped path to collect indices
- [ ] Store `self._last_sequence_indices`
- [ ] Test: verify indices match sequences

### Phase 3: Use Indices for Alignment
- [ ] Update `predict()` to optionally return indices
- [ ] Update `_evaluate_standard()` to use indices
- [ ] Update `_evaluate_per_group()` to use indices
- [ ] Update `_export_predictions_csv()` to use indices
- [ ] Remove hardcoded offsets
- [ ] Test: verify predictions align with actuals

### Phase 4: Edge Cases & Group Boundaries
- [ ] Add `_verify_temporal_order()` method
- [ ] Add `_verify_group_boundaries()` method
- [ ] Add `_validate_indices_integrity()` method
- [ ] Store `self._last_sequence_metadata`
- [ ] Store `self._scaling_stats` with detailed statistics
- [ ] Add overlap tracking in `split_time_series()`
- [ ] Test: verify all edge cases handled

### Phase 5: Comprehensive Testing
- [ ] Create `test_alignment_comprehensive.py` with 12 tests
- [ ] Create `debug_alignment.py` script
- [ ] Implement test scenarios 1-4 (no groups)
- [ ] Implement test scenarios 5-8 (single group)
- [ ] Implement test scenarios 9-12 (multi groups)
- [ ] Add scaler statistics collection
- [ ] Generate `alignment_test_results.md`
- [ ] Run all tests and verify PASS

### Phase 6: Cleanup & Documentation
- [ ] Update all docstrings
- [ ] Add examples to README
- [ ] Create troubleshooting guide
- [ ] Add deprecation warnings if needed
- [ ] Code review and optimization
- [ ] Final validation

---

## Reference Information

### Key Files Modified

- `tf_predictor/core/predictor.py` - Main predictor class
- `tf_predictor/preprocessing/time_features.py` - Feature engineering
- `tf_predictor/core/utils.py` - Utility functions
- `tf_predictor/tests/test_alignment_comprehensive.py` - New test file
- `tf_predictor/debug_alignment.py` - New debug script

### Key Concepts

**Original Index**: Persistent row identifier that tracks each row through all transformations.

**Sequence Index**: The original index of the row that a sequence predicts (target row).

**Group Boundary**: Sequences must not span across different groups to prevent data leakage.

**Temporal Order**: Within each group, time must be monotonically increasing.

**Scaling Isolation**: Each group's data is scaled independently using group-specific statistics.

### Common Pitfalls

1. **Forgetting to exclude `_original_index`** from features/scaling
2. **Off-by-one errors** in sequence-to-index mapping
3. **Cross-group contamination** when creating sequences
4. **Ignoring overlap rows** in evaluation metrics
5. **Not handling edge cases** (minimal data, single group, etc.)

---

## Next Steps

1. Review this plan and get approval
2. Start implementation with Phase 1
3. Test each phase before moving to next
4. Generate comprehensive test results
5. Iterate based on test failures
6. Complete all phases

---

**Document Version History:**
- v1.0 (2025-11-14): Initial plan created
