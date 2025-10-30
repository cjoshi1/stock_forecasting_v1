# Multi-Horizon Evaluation & Visualization Enhancement Plan

## Overview

This plan outlines enhancements to support **per-horizon evaluation metrics** and **per-horizon visualization** for multi-horizon predictions in both FT-Transformer and CSN-Transformer models.

---

## 📊 Part 1: Multi-Horizon Evaluation Metrics

### Current Behavior
- `evaluate()` method returns a single dict with metrics calculated only on first horizon
- No per-horizon breakdown of performance
- Example current output:
  ```python
  {'MAE': 2.5, 'RMSE': 3.2, 'MAPE': 1.8, 'R2': 0.95, ...}
  ```

### Proposed Behavior

#### For Single-Horizon (prediction_horizon=1)
Return same format as current (backward compatible):
```python
{
    'MAE': 2.5,
    'RMSE': 3.2,
    'MAPE': 1.8,
    'R2': 0.95,
    'Directional_Accuracy': 65.0
}
```

#### For Multi-Horizon (prediction_horizon=3)
Return nested dict with overall + per-horizon metrics:
```python
{
    'overall': {
        'MAE': 3.2,      # Average across all horizons
        'RMSE': 4.1,
        'MAPE': 2.1,
        'R2': 0.92,
        'Directional_Accuracy': 62.0
    },
    'horizon_1': {
        'MAE': 2.5,      # Metrics for t+1 predictions
        'RMSE': 3.2,
        'MAPE': 1.8,
        'R2': 0.95,
        'Directional_Accuracy': 65.0
    },
    'horizon_2': {
        'MAE': 3.1,      # Metrics for t+2 predictions
        'RMSE': 4.0,
        'MAPE': 2.0,
        'R2': 0.92,
        'Directional_Accuracy': 61.0
    },
    'horizon_3': {
        'MAE': 4.0,      # Metrics for t+3 predictions
        'RMSE': 5.1,
        'MAPE': 2.5,
        'R2': 0.88,
        'Directional_Accuracy': 60.0
    }
}
```

### Implementation Changes

#### File 1: `tf_predictor/core/utils.py`
**Add new function:**
```python
def calculate_metrics_multi_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prediction_horizon: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for multi-horizon predictions.

    Args:
        y_true: Actual values (1D array, aligned with first horizon)
        y_pred: Predicted values (2D array: n_samples x horizons)
        prediction_horizon: Number of prediction horizons

    Returns:
        Nested dict with overall and per-horizon metrics
    """
    # Calculate per-horizon metrics
    # Calculate overall metrics (average across horizons)
    # Return nested structure
```

#### File 2: `tf_predictor/core/predictor.py`
**Modify `evaluate()` method:**
```python
def evaluate(self, df: pd.DataFrame) -> Dict:
    """Enhanced to return per-horizon metrics for multi-horizon models."""
    # ... existing code ...
    predictions = self.predict(df)

    if self.prediction_horizon == 1:
        # Single-horizon: return simple dict (backward compatible)
        return calculate_metrics(actual, predictions)
    else:
        # Multi-horizon: return nested dict with per-horizon metrics
        return calculate_metrics_multi_horizon(
            actual, predictions, self.prediction_horizon
        )
```

---

## 📈 Part 2: Multi-Horizon Visualization

### Current Behavior
- Plots only show first horizon predictions
- Single comprehensive plot with 4 subplots (train time series, train scatter, test time series, test scatter)

### Proposed Behavior

#### For Single-Horizon (prediction_horizon=1)
Same as current (no changes needed)

#### For Multi-Horizon (prediction_horizon=3)
Generate **separate comprehensive plot for each horizon** plus an **overview comparison plot**:

1. **comprehensive_predictions_h1.png** - Horizon 1 (t+1) predictions
   - Train time series
   - Train scatter
   - Test time series
   - Test scatter
   - Annotated with horizon-specific MAPE

2. **comprehensive_predictions_h2.png** - Horizon 2 (t+2) predictions
   - Same 4-subplot layout for t+2 predictions

3. **comprehensive_predictions_h3.png** - Horizon 3 (t+3) predictions
   - Same 4-subplot layout for t+3 predictions

4. **multi_horizon_comparison.png** - NEW overview plot
   - Layout: 2 rows × 2 columns
   - Top row: Train data with all horizons overlaid
   - Bottom row: Test data with all horizons overlaid
   - Left column: Time series
   - Right column: Performance bar charts (MAE/MAPE by horizon)

5. **training_progress.png** - No changes (same as current)

### Visualization Structure

```
outputs/
├── comprehensive_predictions_h1.png    # Horizon 1 detailed
├── comprehensive_predictions_h2.png    # Horizon 2 detailed
├── comprehensive_predictions_h3.png    # Horizon 3 detailed
├── multi_horizon_comparison.png        # Overview comparison
├── training_progress.png               # Training history
└── data/
    └── predictions_TIMESTAMP.csv       # Enhanced CSV with all horizons
```

### Enhanced CSV Export

#### Current Format:
```csv
date,actual,predicted,dataset,error_abs,error_pct
2020-01-08,100.5,101.2,train,0.7,0.7
```

#### Proposed Multi-Horizon Format:
```csv
date,actual,pred_h1,pred_h2,pred_h3,dataset,error_h1,error_h2,error_h3,mape_h1,mape_h2,mape_h3
2020-01-08,100.5,101.2,102.0,103.1,train,0.7,1.5,2.6,0.7,1.5,2.6
```

### Implementation Changes

#### File 3: `stock_forecasting/visualization/stock_charts.py`

**Modified function:**
```python
def create_comprehensive_plots(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "outputs"
) -> Dict[str, str]:
    """
    Enhanced to generate per-horizon plots for multi-horizon models.

    Returns dict with all saved plot paths:
    - For single-horizon: same as before
    - For multi-horizon: includes h1, h2, h3, and comparison plots
    """
```

**New function:**
```python
def create_horizon_plot(
    train_actual: np.ndarray,
    train_pred: np.ndarray,
    test_actual: np.ndarray,
    test_pred: np.ndarray,
    horizon_num: int,
    output_path: Path
) -> str:
    """
    Create comprehensive 4-subplot plot for a specific horizon.

    Args:
        train_actual: Training actual values
        train_pred: Training predictions for this horizon
        test_actual: Test actual values
        test_pred: Test predictions for this horizon
        horizon_num: Horizon number (1, 2, 3, ...)
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
```

**New function:**
```python
def create_multi_horizon_comparison(
    train_actual: np.ndarray,
    train_preds: np.ndarray,  # Shape: (n_samples, horizons)
    test_actual: np.ndarray,
    test_preds: np.ndarray,   # Shape: (n_samples, horizons)
    metrics: Dict,            # Multi-horizon metrics dict
    output_path: Path
) -> str:
    """
    Create overview comparison plot showing all horizons together.

    Layout:
    ┌─────────────────────┬─────────────────────┐
    │ Train Time Series   │ Train Performance   │
    │ (all horizons)      │ (MAE/MAPE bar chart)│
    ├─────────────────────┼─────────────────────┤
    │ Test Time Series    │ Test Performance    │
    │ (all horizons)      │ (MAE/MAPE bar chart)│
    └─────────────────────┴─────────────────────┘
    """
```

**Modified function:**
```python
def export_predictions_csv(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> str:
    """
    Enhanced to export all horizons for multi-horizon models.

    Detects prediction_horizon and adjusts columns accordingly.
    """
```

#### File 4: `stock_forecasting/main.py`

**Modified section (around lines 239-283):**
```python
if not args.no_plots:
    print(f"\n📊 Generating comprehensive visualizations...")

    try:
        base_output = "outputs"
        saved_plots = create_comprehensive_plots(model, train_df, test_df, base_output)

        # Get predictions and metrics
        train_predictions = model.predict(train_df)
        test_predictions = model.predict(test_df)

        # Get metrics (now handles multi-horizon automatically)
        train_metrics = model.evaluate(train_df)
        test_metrics = model.evaluate(test_df)

        # Print performance summary
        print_performance_summary(
            model,
            train_metrics,
            test_metrics,
            saved_plots
        )

    except Exception as e:
        print(f"   ⚠️  Error generating plots: {e}")
        import traceback
        traceback.print_exc()
```

**Modified function:**
```python
def print_performance_summary(
    model,
    train_metrics: Dict,
    test_metrics: Dict,
    saved_plots: Dict[str, str]
):
    """
    Enhanced to print per-horizon metrics for multi-horizon models.

    For single-horizon:
    🎯 Final Performance Summary:
       📈 Training: MAPE 2.5%, MAE $3.20
       📊 Test: MAPE 3.1%, MAE $4.50

    For multi-horizon:
    🎯 Final Performance Summary:

       📈 Training Metrics:
       ┌──────────┬─────────┬──────────┬─────────┐
       │ Horizon  │   MAE   │   RMSE   │   MAPE  │
       ├──────────┼─────────┼──────────┼─────────┤
       │ h1 (t+1) │  $2.50  │  $3.20   │  1.8%   │
       │ h2 (t+2) │  $3.10  │  $4.00   │  2.0%   │
       │ h3 (t+3) │  $4.00  │  $5.10   │  2.5%   │
       ├──────────┼─────────┼──────────┼─────────┤
       │ Overall  │  $3.20  │  $4.10   │  2.1%   │
       └──────────┴─────────┴──────────┴─────────┘

       📊 Test Metrics: [similar table]

       📁 Saved Plots:
          ✅ Horizon 1: outputs/comprehensive_predictions_h1.png
          ✅ Horizon 2: outputs/comprehensive_predictions_h2.png
          ✅ Horizon 3: outputs/comprehensive_predictions_h3.png
          ✅ Comparison: outputs/multi_horizon_comparison.png
    """
```

---

## 🔧 Part 3: Additional Utilities

### File 5: `stock_forecasting/visualization/multi_horizon_utils.py` (NEW)

Helper utilities for multi-horizon visualization:

```python
def format_metrics_table(metrics: Dict, dataset_name: str) -> str:
    """Format multi-horizon metrics as ASCII table."""

def calculate_horizon_actuals(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    prediction_horizon: int
) -> np.ndarray:
    """
    Calculate actual values for each prediction horizon.

    For horizon h, the actual value is the target h steps after
    the sequence end.

    Returns: (n_samples, horizons) array
    """

def align_predictions_with_actuals(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Align predictions and actuals to same length."""
```

---

## 🎨 Part 4: Visual Design Specifications

### Per-Horizon Plot Design (comprehensive_predictions_h{N}.png)

Same 2×2 layout as current, but with horizon-specific annotations:

```
┌─────────────────────────────┬─────────────────────────────┐
│  Training Time Series       │  Training Scatter           │
│  Title: "Horizon N (t+N)"   │  Perfect prediction line    │
│  MAPE badge: "h1: 1.8%"     │  Correlation shown          │
├─────────────────────────────┼─────────────────────────────┤
│  Test Time Series           │  Test Scatter               │
│  Title: "Horizon N (t+N)"   │  Perfect prediction line    │
│  MAPE badge: "h1: 2.1%"     │  Correlation shown          │
└─────────────────────────────┴─────────────────────────────┘
```

### Multi-Horizon Comparison Plot Design

```
┌─────────────────────────────┬─────────────────────────────┐
│  Training: All Horizons     │  Training: Performance      │
│  ─────────────────────      │  ─────────────────────      │
│  • Actual (solid black)     │    MAE by Horizon           │
│  • h1 (blue dashed)         │    ┌───┐                    │
│  • h2 (green dotted)        │    │ █ │ h1                 │
│  • h3 (red dash-dot)        │    │ █ │ h2                 │
│                             │    │ █ │ h3                 │
│  Legend in top-right        │    └───┘                    │
│  Grid enabled               │                             │
│                             │    MAPE by Horizon          │
│                             │    (similar bar chart)      │
├─────────────────────────────┼─────────────────────────────┤
│  Test: All Horizons         │  Test: Performance          │
│  (same style as train)      │  (same bar charts)          │
└─────────────────────────────┴─────────────────────────────┘
```

**Color Scheme:**
- Actual: Black solid line
- Horizon 1: Blue (`#2E86DE`) dashed
- Horizon 2: Green (`#10AC84`) dotted
- Horizon 3: Red (`#EE5A6F`) dash-dot
- Horizon 4+: Cycle through matplotlib's tab10 colormap

---

## 📦 Part 5: Implementation Order

### Phase 1: Metrics Enhancement
1. Add `calculate_metrics_multi_horizon()` to `utils.py`
2. Modify `evaluate()` in `predictor.py`
3. Add tests for multi-horizon metrics

### Phase 2: Per-Horizon Plots
1. Create `create_horizon_plot()` in `stock_charts.py`
2. Modify `create_comprehensive_plots()` to generate per-horizon plots
3. Test plot generation

### Phase 3: Comparison Plot
1. Create `create_multi_horizon_comparison()` in `stock_charts.py`
2. Integrate into `create_comprehensive_plots()`
3. Test comparison visualization

### Phase 4: Enhanced CSV Export
1. Modify `export_predictions_csv()` to handle all horizons
2. Test CSV output format

### Phase 5: Main Script Integration
1. Update `main.py` to use new metrics structure
2. Create `print_performance_summary()` with table formatting
3. Test end-to-end workflow

### Phase 6: Documentation & Testing
1. Create comprehensive test suite
2. Update documentation
3. Add usage examples

---

## ✅ Backward Compatibility

**Guaranteed:**
- Single-horizon models work exactly as before
- API signatures remain compatible
- Existing scripts continue to work without changes

**Detection:**
- Automatically detect `prediction_horizon` value
- Branch logic based on single vs multi-horizon
- No user configuration needed

---

## 📊 Example Output

### Console Output (Multi-Horizon)
```
📊 Generating comprehensive visualizations...
   ✅ Horizon 1 plot saved: outputs/comprehensive_predictions_h1.png
   ✅ Horizon 2 plot saved: outputs/comprehensive_predictions_h2.png
   ✅ Horizon 3 plot saved: outputs/comprehensive_predictions_h3.png
   ✅ Comparison plot saved: outputs/multi_horizon_comparison.png
   ✅ Training progress saved: outputs/training_progress.png
   ✅ Predictions CSV saved: outputs/data/predictions_2025-10-12_1530.csv

🎯 Final Performance Summary:

   📈 Training Metrics:
   ┌──────────┬─────────┬──────────┬─────────┬────────┐
   │ Horizon  │   MAE   │   RMSE   │   MAPE  │   R²   │
   ├──────────┼─────────┼──────────┼─────────┼────────┤
   │ h1 (t+1) │  $2.50  │  $3.20   │  1.8%   │  0.95  │
   │ h2 (t+2) │  $3.10  │  $4.00   │  2.0%   │  0.92  │
   │ h3 (t+3) │  $4.00  │  $5.10   │  2.5%   │  0.88  │
   ├──────────┼─────────┼──────────┼─────────┼────────┤
   │ Overall  │  $3.20  │  $4.10   │  2.1%   │  0.92  │
   └──────────┴─────────┴──────────┴─────────┴────────┘

   📊 Test Metrics:
   [similar table for test set]

   📁 Generated 5 plots + 1 CSV file
   🔧 Model: 1,234,567 parameters
   📚 Training: 50 epochs
```

---

## 🚀 Benefits

1. **Better Insight:** See how prediction quality degrades over longer horizons
2. **Model Selection:** Compare models based on specific horizon performance
3. **Debugging:** Identify if model struggles with specific horizons
4. **Publication Ready:** Professional plots for each prediction horizon
5. **Complete Export:** CSV includes all horizons for custom analysis

---

## ⚠️ Considerations

1. **File Count:** Multi-horizon creates more files (3-5 plots vs 2 for single-horizon)
2. **Generation Time:** Slightly longer due to multiple plot generation
3. **Disk Space:** More PNGs and larger CSV files
4. **Actual Value Alignment:** Need to carefully align actual values for each horizon (h2 actual = target 2 steps ahead, etc.)

---

## 🎯 Success Criteria

- ✅ Per-horizon metrics calculated correctly
- ✅ Each horizon has its own detailed plot
- ✅ Comparison plot shows all horizons together
- ✅ CSV export includes all horizon predictions
- ✅ Console output shows comprehensive metrics table
- ✅ Backward compatible with single-horizon models
- ✅ Comprehensive test coverage
- ✅ Clear documentation with examples

---

## 📋 Summary

This plan transforms multi-horizon prediction analysis from "first horizon only" to **complete per-horizon evaluation and visualization**. Users will get:

- **Detailed metrics** for each prediction horizon
- **Individual plots** showing performance at each horizon
- **Comparison plots** showing all horizons together
- **Complete CSV export** with all predictions
- **Professional tables** in console output

All while maintaining **100% backward compatibility** with existing single-horizon models.
