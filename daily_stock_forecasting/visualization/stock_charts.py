#!/usr/bin/env python3
"""
Standard visualization module - integrated into main project workflow.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

def create_multi_horizon_comparison(
    train_actual: np.ndarray,
    train_preds: np.ndarray,  # Shape: (n_samples, horizons)
    test_actual: np.ndarray,
    test_preds: np.ndarray,   # Shape: (n_samples, horizons)
    train_metrics: Dict,      # Multi-horizon metrics dict
    test_metrics: Dict,       # Multi-horizon metrics dict
    output_path: Path
) -> str:
    """
    Create overview comparison plot showing all horizons together.

    Args:
        train_actual: Training actual values
        train_preds: Training predictions (2D: n_samples x horizons)
        test_actual: Test actual values
        test_preds: Test predictions (2D: n_samples x horizons)
        train_metrics: Training metrics dict
        test_metrics: Test metrics dict
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
    prediction_horizon = train_preds.shape[1]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    # Color scheme for horizons
    colors = ['#2E86DE', '#10AC84', '#EE5A6F', '#F79F1F', '#A3CB38']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    # Top-left: Training Time Series with all horizons
    train_days = range(1, len(train_actual) + 1)
    ax1.plot(train_days, train_actual, 'k-', label='Actual', linewidth=2.5, alpha=0.9, zorder=10)

    for h in range(prediction_horizon):
        horizon_num = h + 1
        # Align predictions for this horizon
        train_pred_h = train_preds[:, h]
        train_actual_h = train_actual[h:]
        min_len = min(len(train_actual_h), len(train_pred_h))

        color = colors[h % len(colors)]
        linestyle = linestyles[h % len(linestyles)]

        ax1.plot(train_days[h:h+min_len], train_pred_h[:min_len],
                color=color, linestyle=linestyle, label=f'h{horizon_num} (t+{horizon_num})',
                linewidth=1.8, alpha=0.7)

    ax1.set_xlabel('Day Index', fontsize=11)
    ax1.set_ylabel('Close Price ($)', fontsize=11)
    ax1.set_title('Training: All Prediction Horizons', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # Top-right: Training Performance Bar Charts
    horizon_labels = [f'h{i+1}' for i in range(prediction_horizon)]
    mae_values = [train_metrics[f'horizon_{i+1}']['MAE'] for i in range(prediction_horizon)]
    mape_values = [train_metrics[f'horizon_{i+1}']['MAPE'] for i in range(prediction_horizon)]

    x = np.arange(len(horizon_labels))
    width = 0.35

    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width/2, mae_values, width, label='MAE ($)', color='#3498db', alpha=0.7)
    bars2 = ax2_twin.bar(x + width/2, mape_values, width, label='MAPE (%)', color='#e74c3c', alpha=0.7)

    ax2.set_xlabel('Prediction Horizon', fontsize=11)
    ax2.set_ylabel('MAE ($)', fontsize=11, color='#3498db')
    ax2_twin.set_ylabel('MAPE (%)', fontsize=11, color='#e74c3c')
    ax2.set_title('Training: Performance by Horizon', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizon_labels)
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # Bottom-left: Test Time Series with all horizons
    test_days = range(1, len(test_actual) + 1)
    ax3.plot(test_days, test_actual, 'k-', label='Actual', linewidth=2.5, alpha=0.9, zorder=10)

    for h in range(prediction_horizon):
        horizon_num = h + 1
        test_pred_h = test_preds[:, h]
        test_actual_h = test_actual[h:]
        min_len = min(len(test_actual_h), len(test_pred_h))

        color = colors[h % len(colors)]
        linestyle = linestyles[h % len(linestyles)]

        ax3.plot(test_days[h:h+min_len], test_pred_h[:min_len],
                color=color, linestyle=linestyle, label=f'h{horizon_num} (t+{horizon_num})',
                linewidth=1.8, alpha=0.7)

    ax3.set_xlabel('Day Index', fontsize=11)
    ax3.set_ylabel('Close Price ($)', fontsize=11)
    ax3.set_title('Test: All Prediction Horizons', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Test Performance Bar Charts
    mae_values_test = [test_metrics[f'horizon_{i+1}']['MAE'] for i in range(prediction_horizon)]
    mape_values_test = [test_metrics[f'horizon_{i+1}']['MAPE'] for i in range(prediction_horizon)]

    ax4_twin = ax4.twinx()
    bars3 = ax4.bar(x - width/2, mae_values_test, width, label='MAE ($)', color='#3498db', alpha=0.7)
    bars4 = ax4_twin.bar(x + width/2, mape_values_test, width, label='MAPE (%)', color='#e74c3c', alpha=0.7)

    ax4.set_xlabel('Prediction Horizon', fontsize=11)
    ax4.set_ylabel('MAE ($)', fontsize=11, color='#3498db')
    ax4_twin.set_ylabel('MAPE (%)', fontsize=11, color='#e74c3c')
    ax4.set_title('Test: Performance by Horizon', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(horizon_labels)
    ax4.tick_params(axis='y', labelcolor='#3498db')
    ax4_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(output_path)


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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    train_days = range(1, len(train_pred) + 1)
    test_days = range(1, len(test_pred) + 1)

    # Calculate MAPE
    train_mape = np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100
    test_mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

    # Training Time Series
    ax1.plot(train_days, train_actual, 'b-', label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(train_days, train_pred, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Day Index', fontsize=11)
    ax1.set_ylabel('Close Price ($)', fontsize=11)
    ax1.set_title(f'Training Data: Horizon {horizon_num} (t+{horizon_num}) - Time Series', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'h{horizon_num} MAPE: {train_mape:.2f}%', transform=ax1.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    # Training Scatter
    ax2.scatter(train_actual, train_pred, alpha=0.6, s=30, c='blue')
    min_price = min(min(train_actual), min(train_pred))
    max_price = max(max(train_actual), max(train_pred))
    ax2.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, alpha=0.8)
    ax2.set_xlabel('Actual Price ($)', fontsize=11)
    ax2.set_ylabel('Predicted Price ($)', fontsize=11)
    ax2.set_title(f'Training Data: Horizon {horizon_num} (t+{horizon_num}) - Scatter', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Test Time Series
    ax3.plot(test_days, test_actual, 'b-', label='Actual', linewidth=2, alpha=0.8)
    ax3.plot(test_days, test_pred, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Day Index', fontsize=11)
    ax3.set_ylabel('Close Price ($)', fontsize=11)
    ax3.set_title(f'Test Data: Horizon {horizon_num} (t+{horizon_num}) - Time Series', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'h{horizon_num} MAPE: {test_mape:.2f}%', transform=ax3.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    # Test Scatter
    ax4.scatter(test_actual, test_pred, alpha=0.6, s=30, c='green')
    min_price_test = min(min(test_actual), min(test_pred))
    max_price_test = max(max(test_actual), max(test_pred))
    ax4.plot([min_price_test, max_price_test], [min_price_test, max_price_test], 'r--', lw=2, alpha=0.8)
    ax4.set_xlabel('Actual Price ($)', fontsize=11)
    ax4.set_ylabel('Predicted Price ($)', fontsize=11)
    ax4.set_title(f'Test Data: Horizon {horizon_num} (t+{horizon_num}) - Scatter', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(output_path)


def create_comprehensive_plots(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "outputs"
) -> Dict[str, str]:
    """
    Create comprehensive visualization plots with proper alignment and MAPE annotations.

    For single-horizon: generates 1 comprehensive plot
    For multi-horizon: generates separate plots for each horizon + comparison plot

    Args:
        model: Trained StockPredictor model
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save plots

    Returns:
        Dictionary with paths to saved plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get predictions with proper alignment
    train_predictions = model.predict(train_df)
    test_predictions = model.predict(test_df)

    # Get corresponding actual values with proper alignment
    # create_features() does shift(-h) and dropna() which removes last prediction_horizon rows
    # Then create_sequences() removes first sequence_length rows
    # So we need: [sequence_length : -prediction_horizon]
    horizon = model.prediction_horizon
    if horizon == 1:
        train_actual_base = train_df['close'].values[model.sequence_length:-1]
        test_actual_base = test_df['close'].values[model.sequence_length:-1]
    else:
        train_actual_base = train_df['close'].values[model.sequence_length:-horizon]
        test_actual_base = test_df['close'].values[model.sequence_length:-horizon]

    saved_plots = {}

    # Detect if multi-horizon
    is_multi_horizon = len(train_predictions.shape) > 1 and train_predictions.shape[1] > 1

    if not is_multi_horizon:
        # Single-horizon: create one comprehensive plot (arrays are now properly aligned)
        train_actual = train_actual_base
        train_pred = train_predictions
        test_actual = test_actual_base
        test_pred = test_predictions

        # Create single comprehensive plot
        predictions_path = output_path / "comprehensive_predictions.png"
        create_horizon_plot(train_actual, train_pred, test_actual, test_pred, 1, predictions_path)
        print(f"   ✅ Predictions plot saved to: {predictions_path}")
        saved_plots['predictions'] = str(predictions_path)

    else:
        # Multi-horizon: create separate plots for each horizon
        prediction_horizon = train_predictions.shape[1]

        for h in range(prediction_horizon):
            horizon_num = h + 1

            # Get predictions for this horizon
            train_pred_h = train_predictions[:, h]
            test_pred_h = test_predictions[:, h]

            # Get actual values h steps ahead
            # For horizon h (0-indexed), actual values are at position [h:]
            # Arrays are already properly aligned from the base slicing above
            train_actual_h = train_actual_base[h:]
            test_actual_h = test_actual_base[h:]

            # Take minimum to handle horizon offsets (still needed for multi-horizon alignment)
            min_train_len = min(len(train_actual_h), len(train_pred_h))
            min_test_len = min(len(test_actual_h), len(test_pred_h))

            train_actual = train_actual_h[:min_train_len]
            train_pred = train_pred_h[:min_train_len]
            test_actual = test_actual_h[:min_test_len]
            test_pred = test_pred_h[:min_test_len]

            # Create horizon-specific plot
            horizon_plot_path = output_path / f"predictions_horizon_{horizon_num}.png"
            create_horizon_plot(train_actual, train_pred, test_actual, test_pred, horizon_num, horizon_plot_path)
            print(f"   ✅ Horizon {horizon_num} plot saved to: {horizon_plot_path}")
            saved_plots[f'horizon_{horizon_num}'] = str(horizon_plot_path)

        # Create multi-horizon comparison plot
        train_metrics = model.evaluate(train_df)
        test_metrics = model.evaluate(test_df)

        comparison_plot_path = output_path / "multi_horizon_comparison.png"
        create_multi_horizon_comparison(
            train_actual_base, train_predictions,
            test_actual_base, test_predictions,
            train_metrics, test_metrics,
            comparison_plot_path
        )
        print(f"   ✅ Multi-horizon comparison plot saved to: {comparison_plot_path}")
        saved_plots['comparison'] = str(comparison_plot_path)

    # Training Progress Plot (for both single and multi-horizon)
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(model.history['train_loss']) + 1)
    
    plt.plot(epochs, model.history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    if model.history['val_loss'] and len(model.history['val_loss']) > 0:
        val_epochs = range(1, len(model.history['val_loss']) + 1)
        plt.plot(val_epochs, model.history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12) 
    plt.title('Training Progress: Error Reduction Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    final_train_loss = model.history['train_loss'][-1]
    plt.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.6f}', transform=plt.gca().transAxes, 
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    training_path = output_path / "training_progress.png"
    plt.savefig(training_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Training progress plot saved to: {training_path}")
    plt.close()
    saved_plots['training'] = str(training_path)
    
    # 3. Export CSV with date, actual, and predicted values
    csv_path = export_predictions_csv(
        model, train_df, test_df, str(output_path / "data")
    )
    saved_plots['csv'] = csv_path
    
    return saved_plots

def export_predictions_csv(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> str:
    """
    Export predictions to CSV file with date, actual, and predicted values.
    Supports group-based (multi-symbol) and multi-horizon predictions.

    Args:
        model: Trained StockPredictor model
        train_df: Training DataFrame with date column (and optionally group column)
        test_df: Test DataFrame with date column (and optionally group column)
        output_dir: Directory to save CSV file

    Returns:
        Path to saved CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if model uses groups
    has_groups = model.group_column is not None and model.group_column in train_df.columns

    # Get predictions (with group info if available)
    if has_groups:
        train_predictions, train_groups = model.predict(train_df, return_group_info=True)
        test_predictions, test_groups = model.predict(test_df, return_group_info=True)
    else:
        train_predictions = model.predict(train_df)
        test_predictions = model.predict(test_df)
        train_groups = None
        test_groups = None

    # Get actual values and dates (aligned with predictions)
    train_actual_base = train_df['close'].values[model.sequence_length:]
    train_dates = train_df['date'].values[model.sequence_length:]

    test_actual_base = test_df['close'].values[model.sequence_length:]
    test_dates = test_df['date'].values[model.sequence_length:]

    # Detect if multi-horizon
    is_multi_horizon = len(train_predictions.shape) > 1 and train_predictions.shape[1] > 1

    # Create combined dataset
    results = []

    if not is_multi_horizon:
        # Single-horizon: simple format
        min_train_len = min(len(train_actual_base), len(train_predictions), len(train_dates))
        min_test_len = min(len(test_actual_base), len(test_predictions), len(test_dates))

        # Training data
        for i in range(min_train_len):
            row_data = {
                'date': train_dates[i],
                'actual': train_actual_base[i],
                'predicted': train_predictions[i],
                'dataset': 'train',
                'error_abs': abs(train_actual_base[i] - train_predictions[i]),
                'error_pct': abs(train_actual_base[i] - train_predictions[i]) / train_actual_base[i] * 100
            }
            if has_groups:
                row_data[model.group_column] = train_groups[i]
            results.append(row_data)

        # Test data
        for i in range(min_test_len):
            row_data = {
                'date': test_dates[i],
                'actual': test_actual_base[i],
                'predicted': test_predictions[i],
                'dataset': 'test',
                'error_abs': abs(test_actual_base[i] - test_predictions[i]),
                'error_pct': abs(test_actual_base[i] - test_predictions[i]) / test_actual_base[i] * 100
            }
            if has_groups:
                row_data[model.group_column] = test_groups[i]
            results.append(row_data)

        print(f"   Debug alignment check:")
        print(f"   Train: {min_train_len} samples")
        print(f"   Test: {min_test_len} samples")

    else:
        # Multi-horizon: include all horizons
        prediction_horizon = train_predictions.shape[1]

        min_train_len = min(len(train_actual_base), train_predictions.shape[0], len(train_dates))
        min_test_len = min(len(test_actual_base), test_predictions.shape[0], len(test_dates))

        # Training data
        for i in range(min_train_len):
            row_data = {
                'date': train_dates[i],
                'actual': train_actual_base[i],
                'dataset': 'train'
            }

            # Add symbol/group column if present
            if has_groups:
                row_data[model.group_column] = train_groups[i]

            # Add predictions for each horizon
            for h in range(prediction_horizon):
                horizon_num = h + 1
                row_data[f'pred_h{horizon_num}'] = train_predictions[i, h]

                # Calculate errors for each horizon (against actual value h steps ahead)
                if i + h < len(train_actual_base):
                    actual_h_ahead = train_actual_base[i + h]
                    error_abs = abs(actual_h_ahead - train_predictions[i, h])
                    error_pct = (error_abs / actual_h_ahead * 100) if actual_h_ahead != 0 else float('nan')
                    row_data[f'error_h{horizon_num}'] = error_abs
                    row_data[f'mape_h{horizon_num}'] = error_pct
                else:
                    row_data[f'error_h{horizon_num}'] = float('nan')
                    row_data[f'mape_h{horizon_num}'] = float('nan')

            results.append(row_data)

        # Test data
        for i in range(min_test_len):
            row_data = {
                'date': test_dates[i],
                'actual': test_actual_base[i],
                'dataset': 'test'
            }

            # Add symbol/group column if present
            if has_groups:
                row_data[model.group_column] = test_groups[i]

            # Add predictions for each horizon
            for h in range(prediction_horizon):
                horizon_num = h + 1
                row_data[f'pred_h{horizon_num}'] = test_predictions[i, h]

                # Calculate errors for each horizon
                if i + h < len(test_actual_base):
                    actual_h_ahead = test_actual_base[i + h]
                    error_abs = abs(actual_h_ahead - test_predictions[i, h])
                    error_pct = (error_abs / actual_h_ahead * 100) if actual_h_ahead != 0 else float('nan')
                    row_data[f'error_h{horizon_num}'] = error_abs
                    row_data[f'mape_h{horizon_num}'] = error_pct
                else:
                    row_data[f'error_h{horizon_num}'] = float('nan')
                    row_data[f'mape_h{horizon_num}'] = float('nan')

            results.append(row_data)

        print(f"   Debug alignment check:")
        print(f"   Train: {min_train_len} samples × {prediction_horizon} horizons")
        print(f"   Test: {min_test_len} samples × {prediction_horizon} horizons")
    
    # Create DataFrame and sort by date
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('date').reset_index(drop=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    csv_filename = f"predictions_{timestamp}.csv"
    csv_path = output_path / csv_filename
    
    # Save CSV
    results_df.to_csv(csv_path, index=False)
    print(f"   ✅ Predictions CSV saved to: {csv_path}")

    print(f"   📊 Total samples: {len(results_df)}")
    print(f"   🏋️ Training samples: {min_train_len}")
    print(f"   🧪 Test samples: {min_test_len}")

    return str(csv_path)

def _print_per_group_metrics(train_metrics: Dict, test_metrics: Dict, model):
    """
    Print per-group (per-symbol) metrics breakdown.

    Args:
        train_metrics: Training metrics with per-group breakdown
        test_metrics: Test metrics with per-group breakdown
        model: Trained model
    """
    # Get all groups (excluding 'overall')
    groups = sorted([k for k in train_metrics.keys() if k != 'overall'])

    # Check if metrics are also multi-horizon
    overall_metrics = train_metrics['overall']
    is_multi_horizon = isinstance(overall_metrics, dict) and ('horizon_1' in overall_metrics or any(k.startswith('horizon_') for k in overall_metrics.keys()))

    print(f"\n   📊 Per-Symbol Performance Summary:")
    print(f"   Found {len(groups)} symbols: {', '.join(groups)}")

    if not is_multi_horizon:
        # Per-group, single-horizon
        print(f"\n   📈 Training Metrics (by Symbol):")
        print(f"   ┌{'─'*12}┬{'─'*11}┬{'─'*12}┬{'─'*11}┬{'─'*10}┐")
        print(f"   │ {'Symbol':<10} │ {'MAE ($)':<9} │ {'RMSE ($)':<10} │ {'MAPE (%)':<9} │ {'R²':<8} │")
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")

        for group in groups:
            metrics = train_metrics[group]
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            mape = metrics.get('MAPE', 0)
            r2 = metrics.get('R2', 0)
            symbol_short = group[:10] if len(group) > 10 else group
            print(f"   │ {symbol_short:<10} │ ${mae:<8.2f} │ ${rmse:<9.2f} │ {mape:<8.2f}% │ {r2:<8.3f} │")

        # Overall
        overall = train_metrics['overall']
        mae_overall = overall.get('MAE', 0)
        rmse_overall = overall.get('RMSE', 0)
        mape_overall = overall.get('MAPE', 0)
        r2_overall = overall.get('R2', 0)
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
        print(f"   │ {'Overall':<10} │ ${mae_overall:<8.2f} │ ${rmse_overall:<9.2f} │ {mape_overall:<8.2f}% │ {r2_overall:<8.3f} │")
        print(f"   └{'─'*12}┴{'─'*11}┴{'─'*12}┴{'─'*11}┴{'─'*10}┘")

        # Test metrics
        print(f"\n   📊 Test Metrics (by Symbol):")
        print(f"   ┌{'─'*12}┬{'─'*11}┬{'─'*12}┬{'─'*11}┬{'─'*10}┐")
        print(f"   │ {'Symbol':<10} │ {'MAE ($)':<9} │ {'RMSE ($)':<10} │ {'MAPE (%)':<9} │ {'R²':<8} │")
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")

        for group in groups:
            metrics = test_metrics[group]
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            mape = metrics.get('MAPE', 0)
            r2 = metrics.get('R2', 0)
            symbol_short = group[:10] if len(group) > 10 else group
            print(f"   │ {symbol_short:<10} │ ${mae:<8.2f} │ ${rmse:<9.2f} │ {mape:<8.2f}% │ {r2:<8.3f} │")

        # Overall
        overall_test = test_metrics['overall']
        mae_overall_test = overall_test.get('MAE', 0)
        rmse_overall_test = overall_test.get('RMSE', 0)
        mape_overall_test = overall_test.get('MAPE', 0)
        r2_overall_test = overall_test.get('R2', 0)
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
        print(f"   │ {'Overall':<10} │ ${mae_overall_test:<8.2f} │ ${rmse_overall_test:<9.2f} │ {mape_overall_test:<8.2f}% │ {r2_overall_test:<8.3f} │")
        print(f"   └{'─'*12}┴{'─'*11}┴{'─'*12}┴{'─'*11}┴{'─'*10}┘")

    else:
        # Per-group, multi-horizon - print overall metrics then per-symbol summary
        print(f"\n   📈 Overall Training Metrics (All Symbols Combined):")
        overall_train = train_metrics['overall']
        prediction_horizon = len([k for k in overall_train.keys() if k.startswith('horizon_')])

        print(f"   ┌{'─'*12}┬{'─'*11}┬{'─'*12}┬{'─'*11}┬{'─'*10}┐")
        print(f"   │ {'Horizon':<10} │ {'MAE ($)':<9} │ {'RMSE ($)':<10} │ {'MAPE (%)':<9} │ {'R²':<8} │")
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")

        for h in range(1, prediction_horizon + 1):
            metrics = overall_train[f'horizon_{h}']
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            mape = metrics.get('MAPE', 0)
            r2 = metrics.get('R2', 0)
            print(f"   │ h{h} (t+{h}){'':<4} │ ${mae:<8.2f} │ ${rmse:<9.2f} │ {mape:<8.2f}% │ {r2:<8.3f} │")

        overall_avg = overall_train['overall']
        mae_avg = overall_avg.get('MAE', 0)
        rmse_avg = overall_avg.get('RMSE', 0)
        mape_avg = overall_avg.get('MAPE', 0)
        r2_avg = overall_avg.get('R2', 0)
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
        print(f"   │ {'Overall':<10} │ ${mae_avg:<8.2f} │ ${rmse_avg:<9.2f} │ {mape_avg:<8.2f}% │ {r2_avg:<8.3f} │")
        print(f"   └{'─'*12}┴{'─'*11}┴{'─'*12}┴{'─'*11}┴{'─'*10}┘")

        # Per-symbol summary (show overall MAPE per symbol)
        print(f"\n   📈 Training MAPE by Symbol:")
        print(f"   ┌{'─'*12}┬{'─'*50}┐")
        print(f"   │ {'Symbol':<10} │ {'MAPE (%) per Horizon':<48} │")
        print(f"   ├{'─'*12}┼{'─'*50}┤")

        for group in groups:
            group_metrics = train_metrics[group]
            mape_str = ""
            for h in range(1, prediction_horizon + 1):
                mape_h = group_metrics[f'horizon_{h}'].get('MAPE', 0)
                mape_str += f"h{h}:{mape_h:.1f}% "
            symbol_short = group[:10] if len(group) > 10 else group
            print(f"   │ {symbol_short:<10} │ {mape_str:<48} │")

        print(f"   └{'─'*12}┴{'─'*50}┘")

        # Test metrics - overall
        print(f"\n   📊 Overall Test Metrics (All Symbols Combined):")
        overall_test = test_metrics['overall']

        print(f"   ┌{'─'*12}┬{'─'*11}┬{'─'*12}┬{'─'*11}┬{'─'*10}┐")
        print(f"   │ {'Horizon':<10} │ {'MAE ($)':<9} │ {'RMSE ($)':<10} │ {'MAPE (%)':<9} │ {'R²':<8} │")
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")

        for h in range(1, prediction_horizon + 1):
            metrics = overall_test[f'horizon_{h}']
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            mape = metrics.get('MAPE', 0)
            r2 = metrics.get('R2', 0)
            print(f"   │ h{h} (t+{h}){'':<4} │ ${mae:<8.2f} │ ${rmse:<9.2f} │ {mape:<8.2f}% │ {r2:<8.3f} │")

        overall_avg_test = overall_test['overall']
        mae_avg_test = overall_avg_test.get('MAE', 0)
        rmse_avg_test = overall_avg_test.get('RMSE', 0)
        mape_avg_test = overall_avg_test.get('MAPE', 0)
        r2_avg_test = overall_avg_test.get('R2', 0)
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
        print(f"   │ {'Overall':<10} │ ${mae_avg_test:<8.2f} │ ${rmse_avg_test:<9.2f} │ {mape_avg_test:<8.2f}% │ {r2_avg_test:<8.3f} │")
        print(f"   └{'─'*12}┴{'─'*11}┴{'─'*12}┴{'─'*11}┴{'─'*10}┘")

        # Per-symbol test summary
        print(f"\n   📊 Test MAPE by Symbol:")
        print(f"   ┌{'─'*12}┬{'─'*50}┐")
        print(f"   │ {'Symbol':<10} │ {'MAPE (%) per Horizon':<48} │")
        print(f"   ├{'─'*12}┼{'─'*50}┤")

        for group in groups:
            group_metrics = test_metrics[group]
            mape_str = ""
            for h in range(1, prediction_horizon + 1):
                mape_h = group_metrics[f'horizon_{h}'].get('MAPE', 0)
                mape_str += f"h{h}:{mape_h:.1f}% "
            symbol_short = group[:10] if len(group) > 10 else group
            print(f"   │ {symbol_short:<10} │ {mape_str:<48} │")

        print(f"   └{'─'*12}┴{'─'*50}┘")

def print_performance_summary(
    model,
    train_metrics: Dict,
    test_metrics: Dict,
    saved_plots: Dict[str, str]
):
    """
    Print comprehensive performance summary.

    Handles single-horizon, multi-horizon, and per-group metrics.

    Args:
        model: Trained model
        train_metrics: Training metrics dict (simple, nested, or per-group)
        test_metrics: Test metrics dict (simple, nested, or per-group)
        saved_plots: Dict of saved plot paths
    """
    print(f"\n🎯 Final Performance Summary:")

    # Check if per-group metrics (has non-'overall' string keys that aren't 'horizon_X')
    has_group_metrics = any(
        isinstance(k, str) and k != 'overall' and not k.startswith('horizon_')
        for k in train_metrics.keys()
    )

    # Check if multi-horizon ('overall' key with nested dict OR 'horizon_X' keys)
    is_multi_horizon = 'overall' in train_metrics or any(k.startswith('horizon_') for k in train_metrics.keys())

    if has_group_metrics:
        # Per-group metrics display
        _print_per_group_metrics(train_metrics, test_metrics, model)
    elif not is_multi_horizon:
        # Single-horizon: simple display
        train_mape = train_metrics.get('MAPE', 0)
        train_mae = train_metrics.get('MAE', 0)
        test_mape = test_metrics.get('MAPE', 0)
        test_mae = test_metrics.get('MAE', 0)

        print(f"   📈 Training: MAPE {train_mape:.2f}%, MAE ${train_mae:.2f}")
        print(f"   📊 Test: MAPE {test_mape:.2f}%, MAE ${test_mae:.2f}")

    else:
        # Multi-horizon: table display
        prediction_horizon = len([k for k in train_metrics.keys() if k.startswith('horizon_')])

        print(f"\n   📈 Training Metrics:")
        print(f"   ┌{'─'*12}┬{'─'*11}┬{'─'*12}┬{'─'*11}┬{'─'*10}┐")
        print(f"   │ {'Horizon':<10} │ {'MAE ($)':<9} │ {'RMSE ($)':<10} │ {'MAPE (%)':<9} │ {'R²':<8} │")
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")

        for h in range(1, prediction_horizon + 1):
            metrics = train_metrics[f'horizon_{h}']
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            mape = metrics.get('MAPE', 0)
            r2 = metrics.get('R2', 0)
            print(f"   │ h{h} (t+{h}){'':<4} │ ${mae:<8.2f} │ ${rmse:<9.2f} │ {mape:<8.2f}% │ {r2:<8.3f} │")

        # Overall
        overall = train_metrics['overall']
        mae_overall = overall.get('MAE', 0)
        rmse_overall = overall.get('RMSE', 0)
        mape_overall = overall.get('MAPE', 0)
        r2_overall = overall.get('R2', 0)
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
        print(f"   │ {'Overall':<10} │ ${mae_overall:<8.2f} │ ${rmse_overall:<9.2f} │ {mape_overall:<8.2f}% │ {r2_overall:<8.3f} │")
        print(f"   └{'─'*12}┴{'─'*11}┴{'─'*12}┴{'─'*11}┴{'─'*10}┘")

        # Test metrics
        print(f"\n   📊 Test Metrics:")
        print(f"   ┌{'─'*12}┬{'─'*11}┬{'─'*12}┬{'─'*11}┬{'─'*10}┐")
        print(f"   │ {'Horizon':<10} │ {'MAE ($)':<9} │ {'RMSE ($)':<10} │ {'MAPE (%)':<9} │ {'R²':<8} │")
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")

        for h in range(1, prediction_horizon + 1):
            metrics = test_metrics[f'horizon_{h}']
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            mape = metrics.get('MAPE', 0)
            r2 = metrics.get('R2', 0)
            print(f"   │ h{h} (t+{h}){'':<4} │ ${mae:<8.2f} │ ${rmse:<9.2f} │ {mape:<8.2f}% │ {r2:<8.3f} │")

        # Overall
        overall_test = test_metrics['overall']
        mae_overall_test = overall_test.get('MAE', 0)
        rmse_overall_test = overall_test.get('RMSE', 0)
        mape_overall_test = overall_test.get('MAPE', 0)
        r2_overall_test = overall_test.get('R2', 0)
        print(f"   ├{'─'*12}┼{'─'*11}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
        print(f"   │ {'Overall':<10} │ ${mae_overall_test:<8.2f} │ ${rmse_overall_test:<9.2f} │ {mape_overall_test:<8.2f}% │ {r2_overall_test:<8.3f} │")
        print(f"   └{'─'*12}┴{'─'*11}┴{'─'*12}┴{'─'*11}┴{'─'*10}┘")

        # Display saved plots
        print(f"\n   📁 Saved Plots:")
        for h in range(1, prediction_horizon + 1):
            plot_key = f'horizon_{h}'
            if plot_key in saved_plots:
                print(f"      ✅ Horizon {h}: {saved_plots[plot_key]}")
        if 'comparison' in saved_plots:
            print(f"      ✅ Comparison: {saved_plots['comparison']}")

    # Common info
    print(f"   🔧 Model: {sum(p.numel() for p in model.model.parameters()):,} parameters")
    print(f"   📚 Training: {len(model.history['train_loss'])} epochs")