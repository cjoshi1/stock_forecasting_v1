# 🏪 Rossmann Store Sales Forecasting

**Version:** 1.0.0

A specialized time series forecasting module for the [Rossmann Store Sales Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales) built on the TF Predictor framework.

## 🎯 Features

- **Kaggle Competition Ready**: RMSPE evaluation metric, submission file generation
- **Experiment Tracking**: Complete experiment versioning with configs, metrics, and predictions
- **Config-Based Preprocessing**: Version and track different feature engineering strategies
- **Per-Store Scaling**: Independent scaling for each store while training unified model
- **Smart Caching**: Automatically cache preprocessed data for faster iterations
- **Inference Mode**: Predict on test data without target column (Sales)
- **Full TF Predictor Integration**: Access to all model architectures and pooling strategies

## 🚀 Quick Start

### 1. Download Data from Kaggle

First, set up Kaggle API credentials (one-time setup):
```bash
# Download kaggle.json from https://www.kaggle.com/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Download Rossmann data:
```bash
python rossman_forecasting/main.py --download_data
```

### 2. Train with Experiment Tracking (Recommended)

```bash
# Quick test to verify setup
python rossman_forecasting/main.py --experiment_config quick_test

# Full baseline model
python rossman_forecasting/main.py --experiment_config baseline_model
```

Results will be in `rossman_forecasting/experiments/runs/exp_001_<name>/`

**OR** Train with standalone mode:

```bash
python rossman_forecasting/main.py --epochs 50
```

### 3. View Results

**With experiment tracking:**
- All outputs in: `rossman_forecasting/experiments/runs/exp_001_<name>/`
- Summary: `SUMMARY.md`
- Metrics: `metrics.json`
- Predictions: `train_predictions.csv`, `val_predictions.csv`, `test_predictions.csv`
- Kaggle submission: `submission.csv`

**Standalone mode:**
- Model: `rossman_forecasting/models/rossmann_model.pt`
- Predictions: `rossman_forecasting/data/predictions/`
- Kaggle submission: `rossman_forecasting/data/predictions/submission.csv`

## 📊 Project Structure

```
rossman_forecasting/
├── configs/
│   └── preprocessing/
│       ├── baseline.yaml              # Basic features
│       └── competition_enhanced.yaml  # Enhanced competition features
├── experiments/
│   ├── configs/                       # Complete experiment configs
│   │   ├── quick_test.yaml           # Fast test (10 stores, 10 epochs)
│   │   ├── baseline_model.yaml       # Standard baseline
│   │   ├── large_model.yaml          # Large model for best performance
│   │   └── best_kaggle_v1.yaml       # Final tuned submission
│   ├── runs/                         # Experiment outputs (exp_001, exp_002, ...)
│   ├── experiment_results.csv        # Master tracking CSV
│   ├── experiment_tracking.py        # Core tracking utilities
│   ├── compare.py                    # Comparison and analysis tool
│   └── README.md                     # Experiment tracking documentation
├── data/
│   ├── raw/                          # Kaggle data (train.csv, test.csv, store.csv)
│   ├── processed/                    # Cached preprocessed data
│   │   ├── baseline_v1/
│   │   └── competition_v1/
│   └── predictions/                  # Predictions and submissions (standalone mode)
├── preprocessing/
│   ├── rossmann_features.py          # Domain-specific feature engineering
│   └── data_loader.py                # Data loading with caching
├── utils/
│   ├── kaggle_download.py            # Kaggle data downloader
│   ├── metrics.py                    # RMSPE and other metrics
│   └── export.py                     # Submission file generation
├── predictor.py                      # RossmannPredictor class
├── main.py                           # CLI application
├── tests/                            # Test suite
└── README.md
```

## 🔧 Usage Examples

### List Available Preprocessing Configs

```bash
python rossman_forecasting/main.py --list_configs
```

### Train with Custom Preprocessing

```bash
python rossman_forecasting/main.py \
  --preprocessing_config competition_enhanced \
  --epochs 100
```

### Quick Test on Few Stores

```bash
python rossman_forecasting/main.py \
  --max_stores 10 \
  --epochs 20
```

### Full Training with Custom Model

```bash
python rossman_forecasting/main.py \
  --preprocessing_config baseline \
  --d_token 192 \
  --n_layers 4 \
  --n_heads 8 \
  --pooling_type multihead_attention \
  --sequence_length 21 \
  --epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0005
```

### Force Reprocessing

```bash
python rossman_forecasting/main.py \
  --force_preprocess \
  --epochs 50
```

## 📝 CLI Arguments

### Data Options
- `--download_data`: Download data from Kaggle
- `--preprocessing_config`: Config name (default: baseline)
- `--use_cached`: Use cached data if available (default: True)
- `--force_preprocess`: Force reprocessing
- `--max_stores`: Limit to N stores for testing
- `--list_configs`: List available configs

### Model Architecture
- `--sequence_length`: Historical days to use (default: 14)
- `--prediction_horizon`: Steps ahead to predict (default: 1)
- `--model_type`: ft_transformer or csn_transformer
- `--pooling_type`: Pooling strategy (default: multihead_attention)
- `--d_token`: Token dimension (default: 128)
- `--n_layers`: Transformer layers (default: 3)
- `--n_heads`: Attention heads (default: 8)
- `--dropout`: Dropout rate (default: 0.1)
- `--scaler_type`: Scaler type (default: standard)

### Training Options
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 15)

### Data Split
- `--val_ratio`: Validation ratio from train.csv (default: 0.2)

### Output Options
- `--model_path`: Path to save model
- `--export_predictions`: Export predictions to CSV (default: True)
- `--create_submission`: Create Kaggle submission (default: True)
- `--no_save_model`: Don't save the model

## 🎨 Preprocessing Configs

### Baseline Config (`baseline.yaml`)

Basic Rossmann features without advanced engineering:
- Competition distance and months since competition
- Promotion indicators (Promo, Promo2)
- Holiday encoding (StateHoliday, SchoolHoliday)
- Store features (StoreType, Assortment)
- Domain features (SalesPerCustomer)

### Enhanced Config (`competition_enhanced.yaml`)

Adds advanced competition features:
- Binned competition distance
- Distance × MonthsSince interactions

### Creating Custom Configs

Create a new YAML file in `configs/preprocessing/`:

```yaml
version: "my_config_v1"
description: "My custom feature set"

competition:
  enabled: true
  fill_distance_missing: "median"
  calculate_months_since: true

promotion:
  enabled: true
  use_promo2: true
  calculate_months_since_promo2: true
  create_promo_month_indicator: true

holidays:
  enabled: true
  encode_state_holiday: true
  use_school_holiday: true

store:
  enabled: true
  encode_store_type: true
  encode_assortment: true

domain_features:
  sales_per_customer: true
  customers_indicator: true

filtering:
  remove_closed_stores: true
  remove_zero_sales: false

validation:
  strategy: "temporal"
  val_ratio: 0.2

cache:
  enabled: true
  cache_key: "my_config_v1"
```

## 💻 Programmatic API

```python
from rossman_forecasting import RossmannPredictor
from rossman_forecasting.preprocessing import load_and_preprocess_data
from rossman_forecasting.utils import save_kaggle_submission
from tf_predictor.core.utils import split_time_series

# Load and preprocess data
train_processed, test_processed = load_and_preprocess_data(
    config_name='baseline',
    use_cached=True
)

# Split train into train/val
train_df, val_df, _ = split_time_series(
    train_processed,
    val_size=int(len(train_processed) * 0.2 / train_processed['Store'].nunique()),
    group_column='Store',
    time_column='Date',
    sequence_length=14
)

# Initialize predictor
predictor = RossmannPredictor(
    target_column='Sales',
    sequence_length=14,
    prediction_horizon=1,
    group_columns='Store',
    d_token=128,
    n_layers=3,
    n_heads=8
)

# Train
predictor.fit(train_df, val_df, epochs=100, batch_size=64)

# Evaluate with RMSPE
val_metrics = predictor.evaluate(
    val_df,
    export_csv='predictions/val.csv'
)
print(f"Validation RMSPE: {val_metrics['RMSPE']:.4f}")

# Predict on test (inference mode - no Sales column needed)
test_predictions = predictor.predict(test_processed, inference_mode=True)

# Create Kaggle submission
save_kaggle_submission(
    test_processed['Id'].values,
    test_predictions,
    'predictions/submission.csv'
)
```

## 📊 Evaluation Metrics

The module calculates standard metrics plus **RMSPE** (Root Mean Square Percentage Error), the official Kaggle metric:

```
RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))
```

**Note:** Days with Sales = 0 are excluded from RMSPE calculation (Kaggle requirement).

## 🧪 Testing

Run the test suite:

```bash
PYTHONPATH=. python rossman_forecasting/tests/test_rossmann.py
```

## 📦 Data Files

### From Kaggle (downloaded to `data/raw/`):
- `train.csv`: Historical sales (2013-01-01 to 2015-07-31)
- `test.csv`: Test period (2015-08-01 to 2015-09-17)
- `store.csv`: Store metadata

### Generated (in `data/processed/{version}/`):
- `train_processed.csv`: Preprocessed training data
- `test_processed.csv`: Preprocessed test data
- `preprocessing_config.yaml`: Config used for preprocessing

### Predictions (in `data/predictions/`):
- `train_predictions.csv`: Train set predictions with actuals
- `val_predictions.csv`: Validation set predictions with actuals
- `submission.csv`: Kaggle submission file

## 🔍 Feature Engineering

### Rossmann-Specific Features (Added by This Module)

1. **Competition Features**
   - `CompetitionDistance`: Distance to nearest competitor (filled with median)
   - `MonthsSinceCompetition`: Months since competition opened

2. **Promotion Features**
   - `Promo`: Binary indicator for promotion
   - `MonthsSincePromo2`: Months since continuous promotion started
   - `IsPromoMonth`: Is current month in PromoInterval

3. **Holiday Features**
   - `StateHoliday_{a,b,c,0}`: One-hot encoded state holidays
   - `SchoolHoliday`: Binary school holiday indicator

4. **Store Features**
   - `StoreType_{a,b,c,d}`: One-hot encoded store type
   - `Assortment_{a,b,c}`: One-hot encoded assortment level

5. **Domain Features**
   - `SalesPerCustomer`: Average sale per customer
   - `HasCustomerData`: Indicator for customer data availability

### Time-Series Features (Added by TF Predictor Automatically)

- Date features (day, month, year, dayofweek) with cyclical encoding
- Automatic sin/cos encoding for temporal patterns
- Optional lag and rolling features (configurable)

## 🎯 Tips for Best Results

### Model Architecture
- Start with `d_token=128`, `n_layers=3`, `n_heads=8`
- Increase to `d_token=192`, `n_layers=4` for better performance
- Use `pooling_type='multihead_attention'` (default, best performance)

### Sequence Length
- Start with `sequence_length=14` (2 weeks)
- Try `sequence_length=21` (3 weeks) or `sequence_length=30` (1 month)
- Longer sequences may capture weekly/monthly patterns

### Training
- Use `batch_size=64` or `batch_size=128`
- Start with 50-100 epochs
- Learning rate `0.001` works well, try `0.0005` for larger models

### Preprocessing
- Experiment with different configs
- Track results in `experiment_log.csv`
- Use `--force_preprocess` when changing configs

## 🏆 Kaggle Submission

The module automatically generates a submission file in the correct format:

```csv
Id,Sales
1,5263
2,6064
...
```

Upload `rossman_forecasting/data/predictions/submission.csv` to Kaggle to see your score!

## 🧪 Experiment Tracking

The module includes a comprehensive experiment tracking system to help you manage and compare different model configurations.

### Running Experiments

Use complete experiment configs that include preprocessing, model, and training parameters:

```bash
# Quick test (10 stores, 10 epochs)
python rossman_forecasting/main.py --experiment_config quick_test

# Baseline model (all stores, 50 epochs)
python rossman_forecasting/main.py --experiment_config baseline_model

# Large model with enhanced preprocessing
python rossman_forecasting/main.py --experiment_config large_model
```

### Experiment Outputs

Each experiment creates a directory: `experiments/runs/exp_001_<name>/` with:
- `SUMMARY.md`: Human-readable experiment summary
- `metrics.json`: Complete metrics and configuration
- `config_used.yaml`: Exact config used (for reproducibility)
- `model.pt`: Trained model
- `train_predictions.csv`: Train predictions with detailed error analysis
- `val_predictions.csv`: Validation predictions with errors
- `test_predictions.csv`: Test predictions
- `submission.csv`: Kaggle submission file

### Comparing Experiments

View all experiments sorted by RMSPE:
```bash
python rossman_forecasting/experiments/compare.py --show_all
```

Show top 5 experiments:
```bash
python rossman_forecasting/experiments/compare.py --best 5
```

Compare specific experiments:
```bash
python rossman_forecasting/experiments/compare.py --compare exp_001 exp_002 exp_005
```

Show summary statistics:
```bash
python rossman_forecasting/experiments/compare.py --summary
```

### Prediction Files with Error Analysis

Experiment tracking saves predictions with detailed error metrics:

```csv
Store,Date,Actual_Sales,Predicted_Sales,Error,Abs_Error,Pct_Error,RMSPE_Contribution
1,2015-01-01,5263.0,5150.3,-112.7,112.7,-2.14,0.000458
```

This allows deep analysis of where the model performs well or poorly.

### Master Tracking CSV

All experiments are logged to `experiments/experiment_results.csv` with:
- Experiment ID and name
- All metrics (RMSPE, MAE, RMSE, R²)
- Model configuration (architecture, hyperparameters)
- Training info (time, epochs, best epoch)
- Data info (samples, stores, date ranges)
- File paths to all outputs

Perfect for comparing many experiments and identifying the best configurations.

**📖 Full Documentation:** See [experiments/README.md](experiments/README.md) for complete experiment tracking documentation.

## 🐛 Troubleshooting

### "Kaggle API not configured"
Set up Kaggle credentials (see Quick Start)

### "Data not found"
Run with `--download_data` first

### "Insufficient data for splitting"
Reduce `--max_stores` or `--sequence_length`

### Memory issues
- Reduce `--batch_size`
- Reduce `--d_token` or `--n_layers`
- Use `--max_stores` for testing

## 📚 Related Documentation

- [TF Predictor Core](../tf_predictor/README.md)
- [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales)
- [Daily Stock Forecasting Example](../daily_stock_forecasting/README.md)

## 📄 License

This module is part of the TF Predictor framework.
