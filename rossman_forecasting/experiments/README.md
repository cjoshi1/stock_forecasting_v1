# Experiment Tracking System

Comprehensive experiment tracking for Rossmann Store Sales forecasting with reproducibility, comparison, and analysis tools.

## Overview

The experiment tracking system provides:
- **Complete experiment configs** (preprocessing + model + training in single YAML)
- **Automatic experiment ID generation** (exp_001, exp_002, etc.)
- **Per-experiment directories** with all outputs
- **Master CSV tracking** all experiments for comparison
- **Detailed predictions with error analysis**
- **Auto-generated summaries** (SUMMARY.md, metrics.json)
- **Comparison utilities** to analyze results

## Directory Structure

```
rossman_forecasting/experiments/
├── configs/                          # Experiment configuration files
│   ├── quick_test.yaml              # Fast test (10 stores, 10 epochs)
│   ├── baseline_model.yaml          # Standard baseline config
│   ├── large_model.yaml             # Large model for best performance
│   └── best_kaggle_v1.yaml          # Final tuned submission
├── runs/                            # Experiment outputs
│   ├── exp_001_quick_test/
│   │   ├── config_used.yaml         # Config copy for reproducibility
│   │   ├── model.pt                 # Trained model
│   │   ├── train_predictions.csv    # Train predictions with errors
│   │   ├── val_predictions.csv      # Val predictions with errors
│   │   ├── test_predictions.csv     # Test predictions (no actuals)
│   │   ├── submission.csv           # Kaggle submission file
│   │   ├── metrics.json             # Detailed metrics and config
│   │   └── SUMMARY.md               # Human-readable summary
│   └── exp_002_baseline_model/
│       └── ...
├── experiment_results.csv           # Master tracking CSV
├── experiment_tracking.py           # Core tracking utilities
├── compare.py                       # Comparison and analysis tool
└── README.md                        # This file
```

## Quick Start

### 1. Run an Experiment

```bash
# Quick test (10 stores, 10 epochs)
python rossman_forecasting/main.py --experiment_config quick_test

# Baseline model (all stores, 50 epochs)
python rossman_forecasting/main.py --experiment_config baseline_model

# Large model with enhanced preprocessing
python rossman_forecasting/main.py --experiment_config large_model
```

### 2. View Results

After running, experiment outputs will be in:
```
rossman_forecasting/experiments/runs/exp_001_<name>/
```

Key files:
- `SUMMARY.md`: Human-readable overview
- `metrics.json`: Complete metrics and configuration
- `val_predictions.csv`: Validation predictions with detailed error analysis
- `submission.csv`: Kaggle submission file

### 3. Compare Experiments

```bash
# Show all experiments sorted by RMSPE
python rossman_forecasting/experiments/compare.py --show_all

# Show top 5 experiments
python rossman_forecasting/experiments/compare.py --best 5

# Compare specific experiments
python rossman_forecasting/experiments/compare.py --compare exp_001 exp_002 exp_003

# Show summary statistics
python rossman_forecasting/experiments/compare.py --summary
```

## Experiment Configuration

### Config File Structure

Complete experiment configs include preprocessing, model, and training parameters:

```yaml
# experiments/configs/my_experiment.yaml

experiment:
  name: "my_experiment"
  description: "Description of what this experiment tests"

preprocessing:
  config: "baseline"                  # baseline or competition_enhanced
  max_stores: null                    # Limit stores for testing, null = all
  force_preprocess: false             # Force reprocessing data

model:
  model_type: "ft_transformer"        # ft_transformer or csn_transformer
  d_token: 128                        # Token embedding dimension
  n_layers: 3                         # Number of transformer layers
  n_heads: 8                          # Number of attention heads
  pooling_type: "multihead_attention" # Pooling strategy
  sequence_length: 14                 # Historical days to use
  prediction_horizon: 1               # Steps ahead to predict
  dropout: 0.1                        # Dropout rate
  scaler_type: "standard"             # Scaler type

training:
  epochs: 50                          # Training epochs
  batch_size: 64                      # Batch size
  learning_rate: 0.001                # Learning rate
  patience: 15                        # Early stopping patience
  val_ratio: 0.2                      # Validation ratio

output:
  save_model: true                    # Save trained model
  export_predictions: true            # Export predictions
  create_submission: true             # Create Kaggle submission

notes: "Optional notes about this experiment"
```

### Available Configs

#### quick_test.yaml
- **Purpose**: Fast sanity check
- **Stores**: 10 (limited)
- **Epochs**: 10
- **Model**: Small (d_token=64, n_layers=2)
- **Use case**: Testing code changes quickly

#### baseline_model.yaml
- **Purpose**: Standard baseline
- **Stores**: All
- **Epochs**: 50
- **Model**: Medium (d_token=128, n_layers=3)
- **Use case**: Good starting point for experiments

#### large_model.yaml
- **Purpose**: Better performance
- **Stores**: All
- **Epochs**: 100
- **Model**: Large (d_token=192, n_layers=4)
- **Preprocessing**: Enhanced
- **Use case**: Competitive Kaggle score

#### best_kaggle_v1.yaml
- **Purpose**: Final submission tuning
- **Stores**: All
- **Epochs**: 150
- **Model**: Large with longer sequence (30 days)
- **Use case**: Template for final submission

### Creating Custom Configs

1. Copy an existing config from `experiments/configs/`
2. Modify parameters for your experiment
3. Save with descriptive name
4. Run with `--experiment_config <your_config_name>`

## Predictions Output Format

### With Actuals (Train/Val)

```csv
Store,Date,Actual_Sales,Predicted_Sales,Error,Abs_Error,Pct_Error,RMSPE_Contribution
1,2015-01-01,5263.0,5150.3,-112.7,112.7,-2.14,0.000458
1,2015-01-02,6064.0,6200.5,136.5,136.5,2.25,0.000506
...
```

**Columns:**
- `Store`: Store ID
- `Date`: Date
- `Actual_Sales`: Actual sales value
- `Predicted_Sales`: Model prediction
- `Error`: Actual - Predicted
- `Abs_Error`: Absolute error
- `Pct_Error`: Percentage error (%)
- `RMSPE_Contribution`: Squared percentage error (for RMSPE calculation)

### Without Actuals (Test)

```csv
Store,Date,Predicted_Sales
1,2015-08-01,5263.0
1,2015-08-02,6064.0
...
```

## Master Tracking CSV

The master CSV (`experiment_results.csv`) tracks all experiments with key metrics and configurations:

**Key Columns:**
- `experiment_id`: Unique ID (exp_001, exp_002, ...)
- `experiment_name`: Descriptive name
- `timestamp`: When experiment was run
- `val_rmspe`: Validation RMSPE (Kaggle metric)
- `val_mae`, `val_rmse`: Other validation metrics
- `train_rmspe`: Training RMSPE
- `d_token`, `n_layers`, `n_heads`: Model architecture
- `sequence_length`: Historical window size
- `epochs`, `batch_size`, `learning_rate`: Training params
- `preprocessing_version`: Which preprocessing config used
- `training_time_mins`: Training time
- `notes`: Experiment notes

## Comparison Tool

The `compare.py` utility provides several analysis options:

### Show All Experiments

```bash
python rossman_forecasting/experiments/compare.py --show_all
```

Shows table of all experiments with key metrics and config.

### Top N Experiments

```bash
# Top 5 by validation RMSPE (default)
python rossman_forecasting/experiments/compare.py --best 5

# Top 10 by validation MAE
python rossman_forecasting/experiments/compare.py --best 10 --metric val_mae
```

### Compare Specific Experiments

```bash
python rossman_forecasting/experiments/compare.py --compare exp_001 exp_002 exp_005
```

Shows side-by-side comparison of metrics and configurations.

### Summary Statistics

```bash
python rossman_forecasting/experiments/compare.py --summary
```

Shows:
- Total experiments run
- Best RMSPE and which experiment achieved it
- Average metrics across all experiments
- Total training time
- Most common configs

## SUMMARY.md Format

Each experiment generates a human-readable summary:

```markdown
# Experiment: baseline_model

**ID:** exp_001
**Date:** 2024-01-15 14:30:00
**Config:** baseline_model

## Results

### Validation Metrics
- **RMSPE:** 0.1234 ⭐ (Kaggle metric)
- **MAE:** 543.21
- **RMSE:** 789.45
- **R²:** 0.8765

### Training Metrics
- **RMSPE:** 0.1123
- **MAE:** 498.32
- **RMSE:** 712.54

## Configuration

### Preprocessing
- Config: baseline
- Version: baseline_v1

### Model
- Architecture: ft_transformer
- d_token: 128
- Layers: 3
- Heads: 8
- Pooling: multihead_attention
- Sequence: 14 days
- Horizon: 1 step(s)

### Training
- Epochs: 50 (best: 35)
- Batch size: 64
- Learning rate: 0.001
- Time: 25.3 minutes

## Data
- Train samples: 800,000
- Val samples: 200,000
- Test samples: 41,088
- Stores: 1,115

## Files
- Model: `model.pt`
- Train predictions: `train_predictions.csv`
- Val predictions: `val_predictions.csv`
- Test predictions: `test_predictions.csv`
- Kaggle submission: `submission.csv`
- Detailed metrics: `metrics.json`

## Notes
Ready for evaluation. Consider submitting to Kaggle if RMSPE is competitive.
```

## Best Practices

### Experiment Workflow

1. **Start with quick_test** to verify code works
2. **Run baseline_model** to establish baseline performance
3. **Experiment with variations**:
   - Try different preprocessing configs
   - Adjust model architecture (d_token, n_layers)
   - Tune training parameters (learning_rate, batch_size)
   - Vary sequence_length (7, 14, 21, 30 days)
4. **Track everything** - all experiments logged automatically
5. **Compare results** using compare.py
6. **Iterate** based on validation RMSPE
7. **Submit best model** to Kaggle

### Naming Conventions

- Use descriptive experiment names
- Include key changes in notes field
- Keep configs organized by purpose

### Reproducibility

Every experiment saves:
- Complete config used (config_used.yaml)
- Exact preprocessing version
- All hyperparameters
- Random seed (if set)
- Package versions (in metrics.json)

To reproduce: Use the saved `config_used.yaml` from the experiment directory.

## Tips for Best Results

### Model Architecture
- Start with baseline: d_token=128, n_layers=3
- For better performance: d_token=192, n_layers=4
- Use multihead_attention pooling (best results)

### Sequence Length
- Try 14 days (2 weeks) - baseline
- Try 21 days (3 weeks) - captures weekly patterns
- Try 30 days (1 month) - captures monthly patterns

### Training
- Batch size 64 or 128 works well
- Start with 50-100 epochs
- Learning rate 0.001 or 0.0005

### Preprocessing
- Try both baseline and competition_enhanced
- Track which works better for your model

## Troubleshooting

### "Config not found"
Make sure config file exists in `experiments/configs/` with `.yaml` extension.

### "No experiment results found"
No experiments have been run yet. Run one first with `--experiment_config`.

### Memory issues
- Reduce batch_size in config
- Reduce d_token or n_layers
- Use max_stores for testing

### Disk space
Old experiments can be archived or deleted from `experiments/runs/`.
The master CSV will still have the metrics.

## Related Files

- [Main README](../README.md) - Overall module documentation
- [Preprocessing Configs](../configs/preprocessing/) - Data preprocessing configs
- [TF Predictor Core](../../tf_predictor/README.md) - Base framework
