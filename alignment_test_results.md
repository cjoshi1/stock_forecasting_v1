# Alignment Test Results

**Generated:** 2025-11-03 10:27:39

================================================================================
🧪 COMPREHENSIVE ALIGNMENT TESTING
================================================================================

This script will test:
  1. Single-target, multi-horizon (close only)
  2. Multi-target, multi-horizon (close + volume)

Both tests use:
  - 2 groups (AAPL, GOOGL)
  - 10 rows per group
  - Sequence length: 3
  - Prediction horizon: 2
================================================================================
🔍 TEST 1: SINGLE-TARGET ALIGNMENT
================================================================================

Configuration:
  - Targets: close (SINGLE-TARGET)
  - Groups: 2 (AAPL, GOOGL)
  - Rows per group: 10
  - Sequence length: 3
  - Prediction horizon: 2

================================================================================
📊 STEP 0: Raw Input Data
================================================================================
Total rows: 20
Columns (8): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp']

--- AAPL (10 rows) ---
  symbol       date  close  volume
0   AAPL 2024-01-01  100.0    1000
1   AAPL 2024-01-02  101.0    1010
2   AAPL 2024-01-03  102.0    1020
3   AAPL 2024-01-04  103.0    1030
4   AAPL 2024-01-05  104.0    1040
5   AAPL 2024-01-06  105.0    1050
6   AAPL 2024-01-07  106.0    1060
7   AAPL 2024-01-08  107.0    1070
8   AAPL 2024-01-09  108.0    1080
9   AAPL 2024-01-10  109.0    1090

--- GOOGL (10 rows) ---
   symbol       date  close  volume
10  GOOGL 2024-01-01  200.0    2000
11  GOOGL 2024-01-02  201.0    2020
12  GOOGL 2024-01-03  202.0    2040
13  GOOGL 2024-01-04  203.0    2060
14  GOOGL 2024-01-05  204.0    2080
15  GOOGL 2024-01-06  205.0    2100
16  GOOGL 2024-01-07  206.0    2120
17  GOOGL 2024-01-08  207.0    2140
18  GOOGL 2024-01-09  208.0    2160
19  GOOGL 2024-01-10  209.0    2180

================================================================================
Creating Single-Target TimeSeriesPredictor...
================================================================================

✅ Single-target predictor created
   Target columns: ['close']
   Is multi-target: False

--- Running SINGLE-TARGET test ---

================================================================================
STEP 1: _create_base_features()
================================================================================
This step: sorts by group+time, adds date features

💡 NOTES:
   - Synthetic data has 'timestamp' column (copy of 'date' for testing)
   - Sorting uses group_columns: ['symbol']
   - Data is sorted by (group_key, timestamp) for proper temporal order
   - Cyclical encoding: date features → sin/cos pairs (month_sin, month_cos, etc.)
   Sorted data by symbol and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

================================================================================
📊 After _create_base_features()
================================================================================
Total rows: 20
Columns (15): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

--- AAPL (10 rows) ---
  symbol       date   open   high    low  close  volume  month_sin  month_cos
0   AAPL 2024-01-01   99.0  101.0   99.0  100.0    1000        0.5   0.866025
1   AAPL 2024-01-02  100.0  102.0  100.0  101.0    1010        0.5   0.866025
2   AAPL 2024-01-03  101.0  103.0  101.0  102.0    1020        0.5   0.866025
3   AAPL 2024-01-04  102.0  104.0  102.0  103.0    1030        0.5   0.866025
4   AAPL 2024-01-05  103.0  105.0  103.0  104.0    1040        0.5   0.866025
5   AAPL 2024-01-06  104.0  106.0  104.0  105.0    1050        0.5   0.866025
6   AAPL 2024-01-07  105.0  107.0  105.0  106.0    1060        0.5   0.866025
7   AAPL 2024-01-08  106.0  108.0  106.0  107.0    1070        0.5   0.866025
8   AAPL 2024-01-09  107.0  109.0  107.0  108.0    1080        0.5   0.866025
9   AAPL 2024-01-10  108.0  110.0  108.0  109.0    1090        0.5   0.866025

--- GOOGL (10 rows) ---
   symbol       date   open   high    low  close  volume  month_sin  month_cos
10  GOOGL 2024-01-01  199.0  201.0  199.0  200.0    2000        0.5   0.866025
11  GOOGL 2024-01-02  200.0  202.0  200.0  201.0    2020        0.5   0.866025
12  GOOGL 2024-01-03  201.0  203.0  201.0  202.0    2040        0.5   0.866025
13  GOOGL 2024-01-04  202.0  204.0  202.0  203.0    2060        0.5   0.866025
14  GOOGL 2024-01-05  203.0  205.0  203.0  204.0    2080        0.5   0.866025
15  GOOGL 2024-01-06  204.0  206.0  204.0  205.0    2100        0.5   0.866025
16  GOOGL 2024-01-07  205.0  207.0  205.0  206.0    2120        0.5   0.866025
17  GOOGL 2024-01-08  206.0  208.0  206.0  207.0    2140        0.5   0.866025
18  GOOGL 2024-01-09  207.0  209.0  207.0  208.0    2160        0.5   0.866025
19  GOOGL 2024-01-10  208.0  210.0  208.0  209.0    2180        0.5   0.866025

================================================================================
STEP 2: create_shifted_targets()
================================================================================
This step: adds shifted target columns: close_target_h1, close_target_h2
Expected behavior:
  - close_target_h1 = close shifted by -1 (next period's close)
  - close_target_h2 = close shifted by -2 (2 periods ahead close)
  - Last 2 rows per group will have NaN targets (no future data)
  - These NaN rows get dropped
   Created multi-horizon targets for: close
   Prediction horizons: 1 to 2 steps ahead
   Group-based shifting applied using column: ['symbol']
   Remaining samples after shift: 16

================================================================================
📊 After create_shifted_targets()
================================================================================
Total rows: 16
Columns (17): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']

--- AAPL (8 rows) ---
  symbol       date  close  close_target_h1  close_target_h2
0   AAPL 2024-01-01  100.0            101.0            102.0
1   AAPL 2024-01-02  101.0            102.0            103.0
2   AAPL 2024-01-03  102.0            103.0            104.0
3   AAPL 2024-01-04  103.0            104.0            105.0
4   AAPL 2024-01-05  104.0            105.0            106.0
5   AAPL 2024-01-06  105.0            106.0            107.0
6   AAPL 2024-01-07  106.0            107.0            108.0
7   AAPL 2024-01-08  107.0            108.0            109.0

--- GOOGL (8 rows) ---
   symbol       date  close  close_target_h1  close_target_h2
10  GOOGL 2024-01-01  200.0            201.0            202.0
11  GOOGL 2024-01-02  201.0            202.0            203.0
12  GOOGL 2024-01-03  202.0            203.0            204.0
13  GOOGL 2024-01-04  203.0            204.0            205.0
14  GOOGL 2024-01-05  204.0            205.0            206.0
15  GOOGL 2024-01-06  205.0            206.0            207.0
16  GOOGL 2024-01-07  206.0            207.0            208.0
17  GOOGL 2024-01-08  207.0            208.0            209.0

🔑 KEY POINT: This dataframe (df_step2) is what gets stored in _last_processed_df
   It's BEFORE encoding and scaling, so 'symbol' is still a string
   Row count per group: {'AAPL': 8, 'GOOGL': 8}

================================================================================
STEP 3: _encode_categorical_features()
================================================================================
This step: encodes 'symbol' as integers (AAPL=0, GOOGL=1)
   Encoded 'symbol': 2 unique categories
   Categorical cardinalities: {'symbol': 2}

================================================================================
📊 After _encode_categorical_features()
================================================================================
Total rows: 16
Columns (17): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']

--- 0 (8 rows) ---
   symbol       date  close  close_target_h1  close_target_h2
0       0 2024-01-01  100.0            101.0            102.0
1       0 2024-01-02  101.0            102.0            103.0
2       0 2024-01-03  102.0            103.0            104.0
3       0 2024-01-04  103.0            104.0            105.0
4       0 2024-01-05  104.0            105.0            106.0
5       0 2024-01-06  105.0            106.0            107.0
6       0 2024-01-07  106.0            107.0            108.0
7       0 2024-01-08  107.0            108.0            109.0

--- 1 (8 rows) ---
    symbol       date  close  close_target_h1  close_target_h2
10       1 2024-01-01  200.0            201.0            202.0
11       1 2024-01-02  201.0            202.0            203.0
12       1 2024-01-03  202.0            203.0            204.0
13       1 2024-01-04  203.0            204.0            205.0
14       1 2024-01-05  204.0            205.0            206.0
15       1 2024-01-06  205.0            206.0            207.0
16       1 2024-01-07  206.0            207.0            208.0
17       1 2024-01-08  207.0            208.0            209.0

Encoding mapping:
  AAPL -> 0
  GOOGL -> 1

================================================================================
STEP 4: _determine_numerical_columns()
================================================================================
This step: identifies which columns are numerical features (excludes shifted targets)

   Target columns INCLUDED in sequence: ['close']
   Model will have autoregressive information (use_lagged_target_features=True)
   Excluding non-numeric column: date (dtype: datetime64[ns])
   Excluding non-numeric column: timestamp (dtype: datetime64[ns])

📋 Column Classification:
   Numerical feature columns (12): ['close', 'volume', 'open', 'high', 'low', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
   Categorical columns (1): ['symbol']
   Target columns (1): ['close']

💡 NOTES:
   - use_lagged_target_features = True
   - Original target columns are INCLUDED in numerical_columns (autoregressive)
   - Shifted target columns (close_target_h1, etc.) are NEVER in numerical_columns
   - They are scaled separately and used as labels (y), not features (X)

================================================================================
STEP 5: _scale_features_grouped()
================================================================================
This step: scales numerical features AND shifted targets per group
Will scale these shifted targets: ['close_target_h1', 'close_target_h2']
   Scaling features for 2 groups
   Groups: [0 1]
   Group 0: fitted scaler on 8 samples
   Group 1: fitted scaler on 8 samples

📊 Scaler Parameters (per group):

   Group 0 (AAPL):

   Group 1 (GOOGL):

================================================================================
📊 After _scale_features_grouped() [SCALED]
================================================================================
Total rows: 16
Columns (17): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']

--- 0 (8 rows) ---
   symbol       date     close  close_target_h1  close_target_h2
0       0 2024-01-01 -1.527525        -1.527525        -1.527525
1       0 2024-01-02 -1.091089        -1.091089        -1.091089
2       0 2024-01-03 -0.654654        -0.654654        -0.654654
3       0 2024-01-04 -0.218218        -0.218218        -0.218218
4       0 2024-01-05  0.218218         0.218218         0.218218
5       0 2024-01-06  0.654654         0.654654         0.654654
6       0 2024-01-07  1.091089         1.091089         1.091089
7       0 2024-01-08  1.527525         1.527525         1.527525

--- 1 (8 rows) ---
    symbol       date     close  close_target_h1  close_target_h2
10       1 2024-01-01 -1.527525        -1.527525        -1.527525
11       1 2024-01-02 -1.091089        -1.091089        -1.091089
12       1 2024-01-03 -0.654654        -0.654654        -0.654654
13       1 2024-01-04 -0.218218        -0.218218        -0.218218
14       1 2024-01-05  0.218218         0.218218         0.218218
15       1 2024-01-06  0.654654         0.654654         0.654654
16       1 2024-01-07  1.091089         1.091089         1.091089
17       1 2024-01-08  1.527525         1.527525         1.527525

⚠️  Note: Values are now scaled (normalized). Original values are stored in _last_processed_df from Step 2

================================================================================
STEP 6: _prepare_data_grouped() - Create sequences
================================================================================
This step: creates sliding window sequences of length 3

💡 Expected (AFTER FIX):
  - Each group has 8 rows after shifting (10 - 2 dropped)
  - With sequence_length=3, NEW logic creates 8 - 3 + 1 = 6 sequences per group
  - First sequence uses rows [0:3], predicts for row 2 (index 2)
  - Last sequence uses rows [5:8], predicts for row 7 (index 7)
  - Targets are extracted from indices [2:8] (offset = sequence_length - 1 = 2)
  - Total: 6 × 2 groups = 12 sequences
  Created 12 sequences from 2 groups
  X_num shape: torch.Size([12, 3, 12]), X_cat shape: torch.Size([12, 1]), y shape: torch.Size([12, 2])

✅ Sequences created:
   X_num shape: torch.Size([12, 3, 12])
   X_cat shape: torch.Size([12, 1])
   y shape: torch.Size([12, 2])

   Expected: 10 sequences total (5 per group)
   Got: 12 sequences

   Group indices: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

================================================================================
STEP 7: Training model and running ACTUAL evaluation
================================================================================

🔧 Training a simple model for 5 epochs...
   Sorted data by symbol and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
   Created multi-horizon targets for: close
   Prediction horizons: 1 to 2 steps ahead
   Group-based shifting applied using column: ['symbol']
   Remaining samples after shift: 16
   Encoded 'symbol': 2 unique categories
   Categorical cardinalities: {'symbol': 2}

   Target columns INCLUDED in sequence: ['close']
   Model will have autoregressive information (use_lagged_target_features=True)
   Excluding non-numeric column: date (dtype: datetime64[ns])
   Excluding non-numeric column: timestamp (dtype: datetime64[ns])
   Scaling features for 2 groups
   Groups: [0 1]
   Group 0: fitted scaler on 8 samples
   Group 1: fitted scaler on 8 samples
  Created 12 sequences from 2 groups
  X_num shape: torch.Size([12, 3, 12]), X_cat shape: torch.Size([12, 1]), y shape: torch.Size([12, 2])

   Created ft_transformer_cls model:
   - Input: sequences of length 3 with 13 features
   - Output: 2 values (single-target, multi-horizon)
   - Parameters: 26,810
   - Embedding dim: 32
✅ Model trained successfully

================================================================================
STEP 8: Running ACTUAL evaluation to verify alignment
================================================================================

📋 Testing _evaluate_per_group() method...
   This will show exactly what the predictor extracts for actuals

   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

================================================================================
✅ EVALUATION COMPLETED SUCCESSFULLY!
================================================================================

📊 Metrics Structure:
   Top-level keys: ['overall', '0', '1']

   Single-target mode

--- OVERALL METRICS ---

  horizon_1:
    MAE: 0.1439
    MSE: 0.0243
    RMSE: 0.1558
    MAPE: 0.1041
    R2: 1.0000
    Directional_Accuracy: 100.0000

  horizon_2:
    MAE: 0.3258
    MSE: 0.1447
    RMSE: 0.3804
    MAPE: 0.2346
    R2: 0.9999
    Directional_Accuracy: 100.0000

  overall:
    MAE: 0.2348
    MSE: 0.0845
    RMSE: 0.2907
    MAPE: 0.1694
    R2: 1.0000
    Directional_Accuracy: 100.0000

================================================================================
STEP 9: Manual verification of shifted column extraction
================================================================================

🔮 Making predictions to populate internal state...
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
✅ Predictions generated, shape: (12, 2)

📦 Inspecting _last_processed_df (used for evaluation):
   Total rows: 16
   Per group: {'AAPL': 8, 'GOOGL': 8}
   Columns: ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']

   Shifted target columns: ['close_target_h1', 'close_target_h2']

   Extraction offset: 2 (sequence_length - 1)

--- Group 0: AAPL ---
Rows in group: 8

📊 ALIGNMENT TABLE: Date vs Predictions vs Actuals
================================================================================
      Date  close_h1_pred  close_h1_actual  close_h2_pred  close_h2_actual
2024-01-03     103.174942            103.0     104.013329            104.0
2024-01-04     103.930550            104.0     104.545235            105.0
2024-01-05     104.932968            105.0     105.417343            106.0
2024-01-06     106.194077            106.0     106.721230            107.0
2024-01-07     107.242249            107.0     107.859543            108.0
2024-01-08     107.853523            108.0     108.424515            109.0
================================================================================

  Number of predictions for this group: 6
  ✅ PERFECT ALIGNMENT: 6 actuals == 6 predictions

--- Group 1: GOOGL ---
Rows in group: 8

📊 ALIGNMENT TABLE: Date vs Predictions vs Actuals
================================================================================
      Date  close_h1_pred  close_h1_actual  close_h2_pred  close_h2_actual
2024-01-03     203.162201            203.0     204.130432            204.0
2024-01-04     203.918961            204.0     204.683685            205.0
2024-01-05     204.917465            205.0     205.581009            206.0
2024-01-06     206.140198            206.0     206.814224            207.0
2024-01-07     207.123108            207.0     207.822235            208.0
2024-01-08     207.757095            208.0     208.365341            209.0
================================================================================

  Number of predictions for this group: 6
  ✅ PERFECT ALIGNMENT: 6 actuals == 6 predictions

================================================================================
📋 SUMMARY - Data Flow Through Pipeline
================================================================================

1. Raw data:        10 rows per group
2. After shifting:   8 rows per group (2 dropped due to no future data)
3. After sequences:  6 predictions per group (NEW: 8 - 3 + 1 = 6)

4. Evaluation extracts actuals from Step 2 (8 rows per group)
   - Applies NEW sequence offset: 8 - (3-1) = 6 actuals per group
   - For each horizon, extract from shifted columns directly
   - Available: 6 actuals, Predictions: 6
   - ✅  PERFECT ALIGNMENT!

🔍 This demonstrates the NEW alignment (FIXED):
   - Predictions count: (rows_after_shift - sequence_length + 1)
   - Actuals extracted from shifted columns with offset = sequence_length - 1
   - Each horizon is independent: close_target_h1, close_target_h2, etc.
   - Actuals count ALWAYS equals predictions count!

💡 Potential issues to investigate:
   1. Is sequence_length offset applied consistently?
   2. Are group boundaries respected (no data leakage)?
   3. Is the multi-horizon actual extraction correct?
   4. Does _last_processed_df match what we expect?

================================================================================
✅ TEST 1 (SINGLE-TARGET) PASSED
================================================================================


================================================================================
🔍 TEST 2: MULTI-TARGET ALIGNMENT
================================================================================

Configuration:
  - Targets: close, volume (MULTI-TARGET)
  - Groups: 2 (AAPL, GOOGL)
  - Rows per group: 10
  - Sequence length: 3
  - Prediction horizon: 2

================================================================================
Creating Multi-Target TimeSeriesPredictor...
================================================================================

✅ Multi-target predictor created
   Target columns: ['close', 'volume']
   Is multi-target: True

--- Running MULTI-TARGET test ---

================================================================================
STEP 1: _create_base_features()
================================================================================
This step: sorts by group+time, adds date features

💡 NOTES:
   - Synthetic data has 'timestamp' column (copy of 'date' for testing)
   - Sorting uses group_columns: ['symbol']
   - Data is sorted by (group_key, timestamp) for proper temporal order
   - Cyclical encoding: date features → sin/cos pairs (month_sin, month_cos, etc.)
   Sorted data by symbol and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

================================================================================
📊 After _create_base_features()
================================================================================
Total rows: 20
Columns (15): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

--- AAPL (10 rows) ---
  symbol       date   open   high    low  close  volume  month_sin  month_cos
0   AAPL 2024-01-01   99.0  101.0   99.0  100.0    1000        0.5   0.866025
1   AAPL 2024-01-02  100.0  102.0  100.0  101.0    1010        0.5   0.866025
2   AAPL 2024-01-03  101.0  103.0  101.0  102.0    1020        0.5   0.866025
3   AAPL 2024-01-04  102.0  104.0  102.0  103.0    1030        0.5   0.866025
4   AAPL 2024-01-05  103.0  105.0  103.0  104.0    1040        0.5   0.866025
5   AAPL 2024-01-06  104.0  106.0  104.0  105.0    1050        0.5   0.866025
6   AAPL 2024-01-07  105.0  107.0  105.0  106.0    1060        0.5   0.866025
7   AAPL 2024-01-08  106.0  108.0  106.0  107.0    1070        0.5   0.866025
8   AAPL 2024-01-09  107.0  109.0  107.0  108.0    1080        0.5   0.866025
9   AAPL 2024-01-10  108.0  110.0  108.0  109.0    1090        0.5   0.866025

--- GOOGL (10 rows) ---
   symbol       date   open   high    low  close  volume  month_sin  month_cos
10  GOOGL 2024-01-01  199.0  201.0  199.0  200.0    2000        0.5   0.866025
11  GOOGL 2024-01-02  200.0  202.0  200.0  201.0    2020        0.5   0.866025
12  GOOGL 2024-01-03  201.0  203.0  201.0  202.0    2040        0.5   0.866025
13  GOOGL 2024-01-04  202.0  204.0  202.0  203.0    2060        0.5   0.866025
14  GOOGL 2024-01-05  203.0  205.0  203.0  204.0    2080        0.5   0.866025
15  GOOGL 2024-01-06  204.0  206.0  204.0  205.0    2100        0.5   0.866025
16  GOOGL 2024-01-07  205.0  207.0  205.0  206.0    2120        0.5   0.866025
17  GOOGL 2024-01-08  206.0  208.0  206.0  207.0    2140        0.5   0.866025
18  GOOGL 2024-01-09  207.0  209.0  207.0  208.0    2160        0.5   0.866025
19  GOOGL 2024-01-10  208.0  210.0  208.0  209.0    2180        0.5   0.866025

================================================================================
STEP 2: create_shifted_targets()
================================================================================
This step: adds shifted target columns: close_target_h1, close_target_h2, volume_target_h1, volume_target_h2
Expected behavior:
  - close_target_h1 = close shifted by -1 (next period's close)
  - close_target_h2 = close shifted by -2 (2 periods ahead close)
  - volume_target_h1 = volume shifted by -1 (next period's volume)
  - volume_target_h2 = volume shifted by -2 (2 periods ahead volume)
  - Last 2 rows per group will have NaN targets (no future data)
  - These NaN rows get dropped
   Created multi-horizon targets for: close, volume
   Prediction horizons: 1 to 2 steps ahead
   Group-based shifting applied using column: ['symbol']
   Remaining samples after shift: 16

================================================================================
📊 After create_shifted_targets()
================================================================================
Total rows: 16
Columns (19): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

--- AAPL (8 rows) ---
  symbol       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
0   AAPL 2024-01-01  100.0            101.0            102.0    1000            1010.0            1020.0
1   AAPL 2024-01-02  101.0            102.0            103.0    1010            1020.0            1030.0
2   AAPL 2024-01-03  102.0            103.0            104.0    1020            1030.0            1040.0
3   AAPL 2024-01-04  103.0            104.0            105.0    1030            1040.0            1050.0
4   AAPL 2024-01-05  104.0            105.0            106.0    1040            1050.0            1060.0
5   AAPL 2024-01-06  105.0            106.0            107.0    1050            1060.0            1070.0
6   AAPL 2024-01-07  106.0            107.0            108.0    1060            1070.0            1080.0
7   AAPL 2024-01-08  107.0            108.0            109.0    1070            1080.0            1090.0

--- GOOGL (8 rows) ---
   symbol       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
10  GOOGL 2024-01-01  200.0            201.0            202.0    2000            2020.0            2040.0
11  GOOGL 2024-01-02  201.0            202.0            203.0    2020            2040.0            2060.0
12  GOOGL 2024-01-03  202.0            203.0            204.0    2040            2060.0            2080.0
13  GOOGL 2024-01-04  203.0            204.0            205.0    2060            2080.0            2100.0
14  GOOGL 2024-01-05  204.0            205.0            206.0    2080            2100.0            2120.0
15  GOOGL 2024-01-06  205.0            206.0            207.0    2100            2120.0            2140.0
16  GOOGL 2024-01-07  206.0            207.0            208.0    2120            2140.0            2160.0
17  GOOGL 2024-01-08  207.0            208.0            209.0    2140            2160.0            2180.0

🔑 KEY POINT: This dataframe (df_step2) is what gets stored in _last_processed_df
   It's BEFORE encoding and scaling, so 'symbol' is still a string
   Row count per group: {'AAPL': 8, 'GOOGL': 8}

================================================================================
STEP 3: _encode_categorical_features()
================================================================================
This step: encodes 'symbol' as integers (AAPL=0, GOOGL=1)
   Encoded 'symbol': 2 unique categories
   Categorical cardinalities: {'symbol': 2}

================================================================================
📊 After _encode_categorical_features()
================================================================================
Total rows: 16
Columns (19): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

--- 0 (8 rows) ---
   symbol       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
0       0 2024-01-01  100.0            101.0            102.0    1000            1010.0            1020.0
1       0 2024-01-02  101.0            102.0            103.0    1010            1020.0            1030.0
2       0 2024-01-03  102.0            103.0            104.0    1020            1030.0            1040.0
3       0 2024-01-04  103.0            104.0            105.0    1030            1040.0            1050.0
4       0 2024-01-05  104.0            105.0            106.0    1040            1050.0            1060.0
5       0 2024-01-06  105.0            106.0            107.0    1050            1060.0            1070.0
6       0 2024-01-07  106.0            107.0            108.0    1060            1070.0            1080.0
7       0 2024-01-08  107.0            108.0            109.0    1070            1080.0            1090.0

--- 1 (8 rows) ---
    symbol       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
10       1 2024-01-01  200.0            201.0            202.0    2000            2020.0            2040.0
11       1 2024-01-02  201.0            202.0            203.0    2020            2040.0            2060.0
12       1 2024-01-03  202.0            203.0            204.0    2040            2060.0            2080.0
13       1 2024-01-04  203.0            204.0            205.0    2060            2080.0            2100.0
14       1 2024-01-05  204.0            205.0            206.0    2080            2100.0            2120.0
15       1 2024-01-06  205.0            206.0            207.0    2100            2120.0            2140.0
16       1 2024-01-07  206.0            207.0            208.0    2120            2140.0            2160.0
17       1 2024-01-08  207.0            208.0            209.0    2140            2160.0            2180.0

Encoding mapping:
  AAPL -> 0
  GOOGL -> 1

================================================================================
STEP 4: _determine_numerical_columns()
================================================================================
This step: identifies which columns are numerical features (excludes shifted targets)

   Target columns INCLUDED in sequence: ['close', 'volume']
   Model will have autoregressive information (use_lagged_target_features=True)
   Excluding non-numeric column: date (dtype: datetime64[ns])
   Excluding non-numeric column: timestamp (dtype: datetime64[ns])

📋 Column Classification:
   Numerical feature columns (12): ['close', 'volume', 'open', 'high', 'low', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
   Categorical columns (1): ['symbol']
   Target columns (2): ['close', 'volume']

💡 NOTES:
   - use_lagged_target_features = True
   - Original target columns are INCLUDED in numerical_columns (autoregressive)
   - Shifted target columns (close_target_h1, etc.) are NEVER in numerical_columns
   - They are scaled separately and used as labels (y), not features (X)

================================================================================
STEP 5: _scale_features_grouped()
================================================================================
This step: scales numerical features AND shifted targets per group
Will scale these shifted targets: ['close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']
   Scaling features for 2 groups
   Groups: [0 1]
   Group 0: fitted scaler on 8 samples
   Group 1: fitted scaler on 8 samples

📊 Scaler Parameters (per group):

   Group 0 (AAPL):

   Group 1 (GOOGL):

================================================================================
📊 After _scale_features_grouped() [SCALED]
================================================================================
Total rows: 16
Columns (19): ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

--- 0 (8 rows) ---
   symbol       date     close  close_target_h1  close_target_h2    volume  volume_target_h1  volume_target_h2
0       0 2024-01-01 -1.527525        -1.527525        -1.527525 -1.527525         -1.527525         -1.527525
1       0 2024-01-02 -1.091089        -1.091089        -1.091089 -1.091089         -1.091089         -1.091089
2       0 2024-01-03 -0.654654        -0.654654        -0.654654 -0.654654         -0.654654         -0.654654
3       0 2024-01-04 -0.218218        -0.218218        -0.218218 -0.218218         -0.218218         -0.218218
4       0 2024-01-05  0.218218         0.218218         0.218218  0.218218          0.218218          0.218218
5       0 2024-01-06  0.654654         0.654654         0.654654  0.654654          0.654654          0.654654
6       0 2024-01-07  1.091089         1.091089         1.091089  1.091089          1.091089          1.091089
7       0 2024-01-08  1.527525         1.527525         1.527525  1.527525          1.527525          1.527525

--- 1 (8 rows) ---
    symbol       date     close  close_target_h1  close_target_h2    volume  volume_target_h1  volume_target_h2
10       1 2024-01-01 -1.527525        -1.527525        -1.527525 -1.527525         -1.527525         -1.527525
11       1 2024-01-02 -1.091089        -1.091089        -1.091089 -1.091089         -1.091089         -1.091089
12       1 2024-01-03 -0.654654        -0.654654        -0.654654 -0.654654         -0.654654         -0.654654
13       1 2024-01-04 -0.218218        -0.218218        -0.218218 -0.218218         -0.218218         -0.218218
14       1 2024-01-05  0.218218         0.218218         0.218218  0.218218          0.218218          0.218218
15       1 2024-01-06  0.654654         0.654654         0.654654  0.654654          0.654654          0.654654
16       1 2024-01-07  1.091089         1.091089         1.091089  1.091089          1.091089          1.091089
17       1 2024-01-08  1.527525         1.527525         1.527525  1.527525          1.527525          1.527525

⚠️  Note: Values are now scaled (normalized). Original values are stored in _last_processed_df from Step 2

================================================================================
STEP 6: _prepare_data_grouped() - Create sequences
================================================================================
This step: creates sliding window sequences of length 3

💡 Expected (AFTER FIX):
  - Each group has 8 rows after shifting (10 - 2 dropped)
  - With sequence_length=3, NEW logic creates 8 - 3 + 1 = 6 sequences per group
  - First sequence uses rows [0:3], predicts for row 2 (index 2)
  - Last sequence uses rows [5:8], predicts for row 7 (index 7)
  - Targets are extracted from indices [2:8] (offset = sequence_length - 1 = 2)
  - Total: 6 × 2 groups = 12 sequences
  Created 12 sequences from 2 groups
  X_num shape: torch.Size([12, 3, 12]), X_cat shape: torch.Size([12, 1]), y shape: torch.Size([12, 4])

✅ Sequences created:
   X_num shape: torch.Size([12, 3, 12])
   X_cat shape: torch.Size([12, 1])
   y shape: torch.Size([12, 4])

   Expected: 10 sequences total (5 per group)
   Got: 12 sequences

   Group indices: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

================================================================================
STEP 7: Training model and running ACTUAL evaluation
================================================================================

🔧 Training a simple model for 5 epochs...
   Sorted data by symbol and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
   Created multi-horizon targets for: close, volume
   Prediction horizons: 1 to 2 steps ahead
   Group-based shifting applied using column: ['symbol']
   Remaining samples after shift: 16
   Encoded 'symbol': 2 unique categories
   Categorical cardinalities: {'symbol': 2}

   Target columns INCLUDED in sequence: ['close', 'volume']
   Model will have autoregressive information (use_lagged_target_features=True)
   Excluding non-numeric column: date (dtype: datetime64[ns])
   Excluding non-numeric column: timestamp (dtype: datetime64[ns])
   Scaling features for 2 groups
   Groups: [0 1]
   Group 0: fitted scaler on 8 samples
   Group 1: fitted scaler on 8 samples
  Created 12 sequences from 2 groups
  X_num shape: torch.Size([12, 3, 12]), X_cat shape: torch.Size([12, 1]), y shape: torch.Size([12, 4])

   Created ft_transformer_cls model:
   - Input: sequences of length 3 with 13 features
   - Output: 4 values (multi-target, multi-horizon)
   - Parameters: 26,876
   - Embedding dim: 32
✅ Model trained successfully

================================================================================
STEP 8: Running ACTUAL evaluation to verify alignment
================================================================================

📋 Testing _evaluate_per_group() method...
   This will show exactly what the predictor extracts for actuals

   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

================================================================================
✅ EVALUATION COMPLETED SUCCESSFULLY!
================================================================================

📊 Metrics Structure:
   Top-level keys: ['0', '1', 'overall']

   Multi-target mode: 2 targets

================================================================================
OVERALL METRICS (All Groups Combined)
================================================================================

--- Target: close ---

  horizon_1:
    MAE: 0.3339
    MSE: 0.1276
    RMSE: 0.3571
    MAPE: 0.2457
    R2: 0.9999
    Directional_Accuracy: 100.0000

  horizon_2:
    MAE: 0.2660
    MSE: 0.0978
    RMSE: 0.3128
    MAPE: 0.1890
    R2: 1.0000
    Directional_Accuracy: 100.0000

  overall (all horizons):
    MAE: 0.2999
    MSE: 0.1127
    RMSE: 0.3357
    MAPE: 0.2173
    R2: 1.0000
    Directional_Accuracy: 100.0000

--- Target: volume ---

  horizon_1:
    MAE: 4.0775
    MSE: 31.0157
    RMSE: 5.5692
    MAPE: 0.2522
    R2: 0.9999
    Directional_Accuracy: 100.0000

  horizon_2:
    MAE: 4.7198
    MSE: 30.3165
    RMSE: 5.5060
    MAPE: 0.2868
    R2: 0.9999
    Directional_Accuracy: 100.0000

  overall (all horizons):
    MAE: 4.3986
    MSE: 30.6661
    RMSE: 5.5377
    MAPE: 0.2695
    R2: 0.9999
    Directional_Accuracy: 100.0000

================================================================================
STEP 9: Manual verification of shifted column extraction
================================================================================

🔮 Making predictions to populate internal state...
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
✅ Predictions generated, shape: N/A

📦 Inspecting _last_processed_df (used for evaluation):
   Total rows: 16
   Per group: {'AAPL': 8, 'GOOGL': 8}
   Columns: ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

   Shifted target columns: ['close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

   Extraction offset: 2 (sequence_length - 1)

--- Group 0: AAPL ---
Rows in group: 8

📊 ALIGNMENT TABLE: Date vs Predictions vs Actuals
================================================================================
      Date  close_h1_pred  close_h1_actual  close_h2_pred  close_h2_actual  volume_h1_pred  volume_h1_actual  volume_h2_pred  volume_h2_actual
2024-01-03     103.243492            103.0     103.986038            104.0     1032.942627            1030.0     1039.126709            1040.0
2024-01-04     104.269348            104.0     104.585083            105.0     1038.943481            1040.0     1046.724976            1050.0
2024-01-05     105.460808            105.0     105.670898            106.0     1049.419678            1050.0     1059.592896            1060.0
2024-01-06     106.600548            106.0     106.852921            107.0     1059.406860            1060.0     1074.963745            1070.0
2024-01-07     107.292084            107.0     107.804199            108.0     1067.203857            1070.0     1084.658325            1080.0
2024-01-08     107.693527            108.0     108.496086            109.0     1072.529541            1080.0     1087.278442            1090.0
================================================================================

  Number of predictions for this group: 6
  ✅ PERFECT ALIGNMENT: 6 actuals == 6 predictions

--- Group 1: GOOGL ---
Rows in group: 8

📊 ALIGNMENT TABLE: Date vs Predictions vs Actuals
================================================================================
      Date  close_h1_pred  close_h1_actual  close_h2_pred  close_h2_actual  volume_h1_pred  volume_h1_actual  volume_h2_pred  volume_h2_actual
2024-01-03     203.199112            203.0     203.958206            204.0     2064.623779            2060.0     2075.967041            2080.0
2024-01-04     204.202484            204.0     204.545959            205.0     2076.270020            2080.0     2090.933350            2100.0
2024-01-05     205.354034            205.0     205.650589            206.0     2097.194824            2100.0     2116.872803            2120.0
2024-01-06     206.471085            206.0     206.872086            207.0     2118.147217            2120.0     2148.776367            2140.0
2024-01-07     207.173203            207.0     207.842300            208.0     2134.333496            2140.0     2168.877197            2160.0
2024-01-08     207.565643            208.0     208.543884            209.0     2145.187988            2160.0     2174.141602            2180.0
================================================================================

  Number of predictions for this group: 6
  ✅ PERFECT ALIGNMENT: 6 actuals == 6 predictions

================================================================================
📋 SUMMARY - Data Flow Through Pipeline
================================================================================

1. Raw data:        10 rows per group
2. After shifting:   8 rows per group (2 dropped due to no future data)
3. After sequences:  6 predictions per group (NEW: 8 - 3 + 1 = 6)

4. Evaluation extracts actuals from Step 2 (8 rows per group)
   - Applies NEW sequence offset: 8 - (3-1) = 6 actuals per group
   - For each horizon, extract from shifted columns directly
   - Available: 6 actuals, Predictions: 6
   - ✅  PERFECT ALIGNMENT!

🔍 This demonstrates the NEW alignment (FIXED):
   - Predictions count: (rows_after_shift - sequence_length + 1)
   - Actuals extracted from shifted columns with offset = sequence_length - 1
   - Each horizon is independent: close_target_h1, close_target_h2, etc.
   - Actuals count ALWAYS equals predictions count!

💡 Potential issues to investigate:
   1. Is sequence_length offset applied consistently?
   2. Are group boundaries respected (no data leakage)?
   3. Is the multi-horizon actual extraction correct?
   4. Does _last_processed_df match what we expect?

================================================================================
✅ TEST 2 (MULTI-TARGET) PASSED
================================================================================


================================================================================
🎯 FINAL SUMMARY
================================================================================

Both tests completed. Review output above for:
  ✓ Shifted target columns created correctly
  ✓ Extraction from shifted columns (not original)
  ✓ Alignment between predictions and actuals
  ✓ Per-group, per-horizon, overall metrics

Key verification points:
  - close_target_h1, close_target_h2 for single-target
  - close_target_h1, close_target_h2, volume_target_h1, volume_target_h2 for multi-target
  - Offset = sequence_length - 1 = 2
  - 6 predictions per group (8 rows after shift - 3 + 1)


---
**Results saved to:** `alignment_test_results.md`
