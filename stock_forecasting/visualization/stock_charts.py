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

def create_comprehensive_plots(
    model,
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    output_dir: str = "outputs"
) -> Dict[str, str]:
    """
    Create comprehensive visualization plots with proper alignment and MAPE annotations.
    
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
    
    # Get corresponding actual values
    train_actual_values = train_df['close'].values[model.sequence_length:]
    test_actual_values = test_df['close'].values[model.sequence_length:]
    
    # Ensure same lengths
    min_train_len = min(len(train_actual_values), len(train_predictions))
    min_test_len = min(len(test_actual_values), len(test_predictions))
    
    train_actual = train_actual_values[:min_train_len]
    train_pred = train_predictions[:min_train_len]
    test_actual = test_actual_values[:min_test_len]
    test_pred = test_predictions[:min_test_len]
    
    # Calculate MAPE
    train_mape = np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100
    test_mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
    
    saved_plots = {}
    
    # 1. Comprehensive Predictions Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    train_days = range(1, len(train_pred) + 1)
    test_days = range(1, len(test_pred) + 1)
    
    # Training Time Series
    ax1.plot(train_days, train_actual, 'b-', label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(train_days, train_pred, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Day Index', fontsize=11)
    ax1.set_ylabel('Close Price ($)', fontsize=11)
    ax1.set_title('Training Data: Actual vs Predicted (Time Series)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Train MAPE: {train_mape:.2f}%', transform=ax1.transAxes, 
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    # Training Scatter
    ax2.scatter(train_actual, train_pred, alpha=0.6, s=30, c='blue')
    min_price = min(min(train_actual), min(train_pred))
    max_price = max(max(train_actual), max(train_pred))
    ax2.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, alpha=0.8)
    ax2.set_xlabel('Actual Price ($)', fontsize=11)
    ax2.set_ylabel('Predicted Price ($)', fontsize=11)
    ax2.set_title('Training Data: Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Test Time Series
    ax3.plot(test_days, test_actual, 'b-', label='Actual', linewidth=2, alpha=0.8)
    ax3.plot(test_days, test_pred, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Day Index', fontsize=11)
    ax3.set_ylabel('Close Price ($)', fontsize=11)
    ax3.set_title('Test Data: Actual vs Predicted (Time Series)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'Test MAPE: {test_mape:.2f}%', transform=ax3.transAxes, 
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    # Test Scatter
    ax4.scatter(test_actual, test_pred, alpha=0.6, s=30, c='green')
    min_price_test = min(min(test_actual), min(test_pred))
    max_price_test = max(max(test_actual), max(test_pred))
    ax4.plot([min_price_test, max_price_test], [min_price_test, max_price_test], 'r--', lw=2, alpha=0.8)
    ax4.set_xlabel('Actual Price ($)', fontsize=11)
    ax4.set_ylabel('Predicted Price ($)', fontsize=11)
    ax4.set_title('Test Data: Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    predictions_path = output_path / "comprehensive_predictions.png"
    plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Predictions plot saved to: {predictions_path}")
    plt.close()
    saved_plots['predictions'] = str(predictions_path)
    
    # 2. Training Progress Plot
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
    
    Args:
        model: Trained StockPredictor model
        train_df: Training DataFrame with date column
        test_df: Test DataFrame with date column
        output_dir: Directory to save CSV file
        
    Returns:
        Path to saved CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    train_predictions = model.predict(train_df)
    test_predictions = model.predict(test_df)
    
    # Get actual values and dates (aligned with predictions)
    # For sequence model: we predict the target at position i using sequence [i-seq_len:i]
    # So predictions align with targets starting from sequence_length index
    train_actual_values = train_df['close'].values[model.sequence_length:]
    train_dates = train_df['date'].values[model.sequence_length:]
    
    test_actual_values = test_df['close'].values[model.sequence_length:]  
    test_dates = test_df['date'].values[model.sequence_length:]
    
    # Debug alignment
    print(f"   Debug alignment check:")
    print(f"   Train: {len(train_predictions)} predictions, {len(train_actual_values)} actuals, {len(train_dates)} dates")
    print(f"   Test: {len(test_predictions)} predictions, {len(test_actual_values)} actuals, {len(test_dates)} dates")
    print(f"   Train first prediction date: {train_dates[0] if len(train_dates) > 0 else 'None'}")
    print(f"   Test first prediction date: {test_dates[0] if len(test_dates) > 0 else 'None'}")
    
    # Ensure same lengths
    min_train_len = min(len(train_actual_values), len(train_predictions), len(train_dates))
    min_test_len = min(len(test_actual_values), len(test_predictions), len(test_dates))
    
    # Create combined dataset
    results = []
    
    # Training data
    for i in range(min_train_len):
        results.append({
            'date': train_dates[i],
            'actual': train_actual_values[i],
            'predicted': train_predictions[i],
            'dataset': 'train',
            'error_abs': abs(train_actual_values[i] - train_predictions[i]),
            'error_pct': abs(train_actual_values[i] - train_predictions[i]) / train_actual_values[i] * 100
        })
    
    # Test data
    for i in range(min_test_len):
        results.append({
            'date': test_dates[i],
            'actual': test_actual_values[i], 
            'predicted': test_predictions[i],
            'dataset': 'test',
            'error_abs': abs(test_actual_values[i] - test_predictions[i]),
            'error_pct': abs(test_actual_values[i] - test_predictions[i]) / test_actual_values[i] * 100
        })
    
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

def print_performance_summary(
    model,
    train_actual: np.ndarray,
    train_pred: np.ndarray, 
    test_actual: np.ndarray,
    test_pred: np.ndarray
):
    """Print comprehensive performance summary."""
    train_mape = np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100
    test_mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
    train_mae = np.mean(np.abs(train_actual - train_pred))
    test_mae = np.mean(np.abs(test_actual - test_pred))
    
    print(f"\n🎯 Final Performance Summary:")
    print(f"   📈 Training: MAPE {train_mape:.2f}%, MAE ${train_mae:.2f}")
    print(f"   📊 Test: MAPE {test_mape:.2f}%, MAE ${test_mae:.2f}")  
    print(f"   🔧 Model: {sum(p.numel() for p in model.model.parameters()):,} parameters")
    print(f"   📚 Training: {len(model.history['train_loss'])} epochs")
    
    return {
        'train_mape': train_mape,
        'test_mape': test_mape,
        'train_mae': train_mae,
        'test_mae': test_mae
    }