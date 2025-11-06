# Alignment Test Results - MULTI-COLUMN GROUPING

**Generated:** 2025-11-06 13:40:12

================================================================================
🧪 COMPREHENSIVE ALIGNMENT TESTING WITH MULTI-COLUMN GROUPING
================================================================================

This script will test:
  1. Single-target, multi-horizon (close only)
  2. Multi-target, multi-horizon (close + volume)

Both tests use:
  - 4 groups with MULTI-COLUMN grouping ['symbol', 'sector']:
    * (AAPL, Tech)
    * (GOOGL, Tech)
    * (MSFT, Consumer)
    * (AMZN, Consumer)
  - 10 rows per group
  - Sequence length: 3
  - Prediction horizon: 2

💡 This test validates the FIX for multi-column grouping bug!
================================================================================
🔍 TEST 1: SINGLE-TARGET ALIGNMENT (MULTI-COLUMN GROUPING)
================================================================================

Configuration:
  - Targets: close (SINGLE-TARGET)
  - Groups: 4 with MULTI-COLUMN grouping ['symbol', 'sector']
    * (AAPL, Tech)
    * (GOOGL, Tech)
    * (MSFT, Consumer)
    * (AMZN, Consumer)
  - Rows per group: 10
  - Sequence length: 3
  - Prediction horizon: 2

================================================================================
📊 STEP 0: Raw Input Data
================================================================================
Total rows: 40
Columns (9): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp']
   symbol    sector       date  close  volume
0    AAPL      Tech 2024-01-01  100.0    1000
1    AAPL      Tech 2024-01-02  101.0    1010
2    AAPL      Tech 2024-01-03  102.0    1020
3    AAPL      Tech 2024-01-04  103.0    1030
4    AAPL      Tech 2024-01-05  104.0    1040
5    AAPL      Tech 2024-01-06  105.0    1050
6    AAPL      Tech 2024-01-07  106.0    1060
7    AAPL      Tech 2024-01-08  107.0    1070
8    AAPL      Tech 2024-01-09  108.0    1080
9    AAPL      Tech 2024-01-10  109.0    1090
10  GOOGL      Tech 2024-01-01  200.0    2000
11  GOOGL      Tech 2024-01-02  201.0    2020
12  GOOGL      Tech 2024-01-03  202.0    2040
13  GOOGL      Tech 2024-01-04  203.0    2060
14  GOOGL      Tech 2024-01-05  204.0    2080
15  GOOGL      Tech 2024-01-06  205.0    2100
16  GOOGL      Tech 2024-01-07  206.0    2120
17  GOOGL      Tech 2024-01-08  207.0    2140
18  GOOGL      Tech 2024-01-09  208.0    2160
19  GOOGL      Tech 2024-01-10  209.0    2180
20   MSFT  Consumer 2024-01-01  300.0    3000
21   MSFT  Consumer 2024-01-02  301.0    3030
22   MSFT  Consumer 2024-01-03  302.0    3060
23   MSFT  Consumer 2024-01-04  303.0    3090
24   MSFT  Consumer 2024-01-05  304.0    3120
25   MSFT  Consumer 2024-01-06  305.0    3150
26   MSFT  Consumer 2024-01-07  306.0    3180
27   MSFT  Consumer 2024-01-08  307.0    3210
28   MSFT  Consumer 2024-01-09  308.0    3240
29   MSFT  Consumer 2024-01-10  309.0    3270
30   AMZN  Consumer 2024-01-01  400.0    4000
31   AMZN  Consumer 2024-01-02  401.0    4040
32   AMZN  Consumer 2024-01-03  402.0    4080
33   AMZN  Consumer 2024-01-04  403.0    4120
34   AMZN  Consumer 2024-01-05  404.0    4160
35   AMZN  Consumer 2024-01-06  405.0    4200
36   AMZN  Consumer 2024-01-07  406.0    4240
37   AMZN  Consumer 2024-01-08  407.0    4280
38   AMZN  Consumer 2024-01-09  408.0    4320
39   AMZN  Consumer 2024-01-10  409.0    4360

================================================================================
Creating Single-Target TimeSeriesPredictor with MULTI-COLUMN grouping...
================================================================================

✅ Single-target predictor created
   Target columns: ['close']
   Group columns: ['symbol', 'sector']
   Is multi-target: False

--- Running SINGLE-TARGET-MULTIGROUP test ---

================================================================================
STEP 1: _create_base_features()
================================================================================
This step: sorts by group+time, adds date features

💡 NOTES:
   - Synthetic data has 'timestamp' column (copy of 'date' for testing)
   - Sorting uses group_columns: ['symbol', 'sector']
   - Data is sorted by (group_key, timestamp) for proper temporal order
   - Cyclical encoding: date features → sin/cos pairs (month_sin, month_cos, etc.)
   Sorted data by symbol + sector and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

================================================================================
📊 After _create_base_features()
================================================================================
Total rows: 40
Columns (16): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
         date   open   high    low  close  volume  month_sin  month_cos
0  2024-01-01   99.0  101.0   99.0  100.0    1000        0.5   0.866025
1  2024-01-02  100.0  102.0  100.0  101.0    1010        0.5   0.866025
2  2024-01-03  101.0  103.0  101.0  102.0    1020        0.5   0.866025
3  2024-01-04  102.0  104.0  102.0  103.0    1030        0.5   0.866025
4  2024-01-05  103.0  105.0  103.0  104.0    1040        0.5   0.866025
5  2024-01-06  104.0  106.0  104.0  105.0    1050        0.5   0.866025
6  2024-01-07  105.0  107.0  105.0  106.0    1060        0.5   0.866025
7  2024-01-08  106.0  108.0  106.0  107.0    1070        0.5   0.866025
8  2024-01-09  107.0  109.0  107.0  108.0    1080        0.5   0.866025
9  2024-01-10  108.0  110.0  108.0  109.0    1090        0.5   0.866025
10 2024-01-01  399.0  401.0  399.0  400.0    4000        0.5   0.866025
11 2024-01-02  400.0  402.0  400.0  401.0    4040        0.5   0.866025
12 2024-01-03  401.0  403.0  401.0  402.0    4080        0.5   0.866025
13 2024-01-04  402.0  404.0  402.0  403.0    4120        0.5   0.866025
14 2024-01-05  403.0  405.0  403.0  404.0    4160        0.5   0.866025
15 2024-01-06  404.0  406.0  404.0  405.0    4200        0.5   0.866025
16 2024-01-07  405.0  407.0  405.0  406.0    4240        0.5   0.866025
17 2024-01-08  406.0  408.0  406.0  407.0    4280        0.5   0.866025
18 2024-01-09  407.0  409.0  407.0  408.0    4320        0.5   0.866025
19 2024-01-10  408.0  410.0  408.0  409.0    4360        0.5   0.866025
20 2024-01-01  199.0  201.0  199.0  200.0    2000        0.5   0.866025
21 2024-01-02  200.0  202.0  200.0  201.0    2020        0.5   0.866025
22 2024-01-03  201.0  203.0  201.0  202.0    2040        0.5   0.866025
23 2024-01-04  202.0  204.0  202.0  203.0    2060        0.5   0.866025
24 2024-01-05  203.0  205.0  203.0  204.0    2080        0.5   0.866025
25 2024-01-06  204.0  206.0  204.0  205.0    2100        0.5   0.866025
26 2024-01-07  205.0  207.0  205.0  206.0    2120        0.5   0.866025
27 2024-01-08  206.0  208.0  206.0  207.0    2140        0.5   0.866025
28 2024-01-09  207.0  209.0  207.0  208.0    2160        0.5   0.866025
29 2024-01-10  208.0  210.0  208.0  209.0    2180        0.5   0.866025
30 2024-01-01  299.0  301.0  299.0  300.0    3000        0.5   0.866025
31 2024-01-02  300.0  302.0  300.0  301.0    3030        0.5   0.866025
32 2024-01-03  301.0  303.0  301.0  302.0    3060        0.5   0.866025
33 2024-01-04  302.0  304.0  302.0  303.0    3090        0.5   0.866025
34 2024-01-05  303.0  305.0  303.0  304.0    3120        0.5   0.866025
35 2024-01-06  304.0  306.0  304.0  305.0    3150        0.5   0.866025
36 2024-01-07  305.0  307.0  305.0  306.0    3180        0.5   0.866025
37 2024-01-08  306.0  308.0  306.0  307.0    3210        0.5   0.866025
38 2024-01-09  307.0  309.0  307.0  308.0    3240        0.5   0.866025
39 2024-01-10  308.0  310.0  308.0  309.0    3270        0.5   0.866025

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
   Group-based shifting applied using column: ['symbol', 'sector']
   Remaining samples after shift: 32

================================================================================
📊 After create_shifted_targets()
================================================================================
Total rows: 32
Columns (18): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']

--- symbol=AAPL + sector=Tech (8 rows) ---
  symbol sector       date  close  close_target_h1  close_target_h2
0   AAPL   Tech 2024-01-01  100.0            101.0            102.0
1   AAPL   Tech 2024-01-02  101.0            102.0            103.0
2   AAPL   Tech 2024-01-03  102.0            103.0            104.0
3   AAPL   Tech 2024-01-04  103.0            104.0            105.0
4   AAPL   Tech 2024-01-05  104.0            105.0            106.0
5   AAPL   Tech 2024-01-06  105.0            106.0            107.0
6   AAPL   Tech 2024-01-07  106.0            107.0            108.0
7   AAPL   Tech 2024-01-08  107.0            108.0            109.0

--- symbol=AMZN + sector=Consumer (8 rows) ---
   symbol    sector       date  close  close_target_h1  close_target_h2
10   AMZN  Consumer 2024-01-01  400.0            401.0            402.0
11   AMZN  Consumer 2024-01-02  401.0            402.0            403.0
12   AMZN  Consumer 2024-01-03  402.0            403.0            404.0
13   AMZN  Consumer 2024-01-04  403.0            404.0            405.0
14   AMZN  Consumer 2024-01-05  404.0            405.0            406.0
15   AMZN  Consumer 2024-01-06  405.0            406.0            407.0
16   AMZN  Consumer 2024-01-07  406.0            407.0            408.0
17   AMZN  Consumer 2024-01-08  407.0            408.0            409.0

--- symbol=GOOGL + sector=Tech (8 rows) ---
   symbol sector       date  close  close_target_h1  close_target_h2
20  GOOGL   Tech 2024-01-01  200.0            201.0            202.0
21  GOOGL   Tech 2024-01-02  201.0            202.0            203.0
22  GOOGL   Tech 2024-01-03  202.0            203.0            204.0
23  GOOGL   Tech 2024-01-04  203.0            204.0            205.0
24  GOOGL   Tech 2024-01-05  204.0            205.0            206.0
25  GOOGL   Tech 2024-01-06  205.0            206.0            207.0
26  GOOGL   Tech 2024-01-07  206.0            207.0            208.0
27  GOOGL   Tech 2024-01-08  207.0            208.0            209.0

--- symbol=MSFT + sector=Consumer (8 rows) ---
   symbol    sector       date  close  close_target_h1  close_target_h2
30   MSFT  Consumer 2024-01-01  300.0            301.0            302.0
31   MSFT  Consumer 2024-01-02  301.0            302.0            303.0
32   MSFT  Consumer 2024-01-03  302.0            303.0            304.0
33   MSFT  Consumer 2024-01-04  303.0            304.0            305.0
34   MSFT  Consumer 2024-01-05  304.0            305.0            306.0
35   MSFT  Consumer 2024-01-06  305.0            306.0            307.0
36   MSFT  Consumer 2024-01-07  306.0            307.0            308.0
37   MSFT  Consumer 2024-01-08  307.0            308.0            309.0

🔑 KEY POINT: This dataframe (df_step2) is what gets stored in _last_processed_df
   It's BEFORE encoding and scaling, so ['symbol', 'sector'] are still strings/original values
   Row count per group: {('AAPL', 'Tech'): 8, ('AMZN', 'Consumer'): 8, ('GOOGL', 'Tech'): 8, ('MSFT', 'Consumer'): 8}

================================================================================
STEP 3: _encode_categorical_features()
================================================================================
This step: encodes 'symbol' as integers (AAPL=0, GOOGL=1)
   Encoded 'symbol': 4 unique categories
   Encoded 'sector': 2 unique categories
   Categorical cardinalities: {'symbol': 4, 'sector': 2}

================================================================================
📊 After _encode_categorical_features()
================================================================================
Total rows: 32
Columns (18): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']

--- symbol=0 + sector=1 (8 rows) ---
   symbol  sector       date  close  close_target_h1  close_target_h2
0       0       1 2024-01-01  100.0            101.0            102.0
1       0       1 2024-01-02  101.0            102.0            103.0
2       0       1 2024-01-03  102.0            103.0            104.0
3       0       1 2024-01-04  103.0            104.0            105.0
4       0       1 2024-01-05  104.0            105.0            106.0
5       0       1 2024-01-06  105.0            106.0            107.0
6       0       1 2024-01-07  106.0            107.0            108.0
7       0       1 2024-01-08  107.0            108.0            109.0

--- symbol=1 + sector=0 (8 rows) ---
    symbol  sector       date  close  close_target_h1  close_target_h2
10       1       0 2024-01-01  400.0            401.0            402.0
11       1       0 2024-01-02  401.0            402.0            403.0
12       1       0 2024-01-03  402.0            403.0            404.0
13       1       0 2024-01-04  403.0            404.0            405.0
14       1       0 2024-01-05  404.0            405.0            406.0
15       1       0 2024-01-06  405.0            406.0            407.0
16       1       0 2024-01-07  406.0            407.0            408.0
17       1       0 2024-01-08  407.0            408.0            409.0

--- symbol=2 + sector=1 (8 rows) ---
    symbol  sector       date  close  close_target_h1  close_target_h2
20       2       1 2024-01-01  200.0            201.0            202.0
21       2       1 2024-01-02  201.0            202.0            203.0
22       2       1 2024-01-03  202.0            203.0            204.0
23       2       1 2024-01-04  203.0            204.0            205.0
24       2       1 2024-01-05  204.0            205.0            206.0
25       2       1 2024-01-06  205.0            206.0            207.0
26       2       1 2024-01-07  206.0            207.0            208.0
27       2       1 2024-01-08  207.0            208.0            209.0

--- symbol=3 + sector=0 (8 rows) ---
    symbol  sector       date  close  close_target_h1  close_target_h2
30       3       0 2024-01-01  300.0            301.0            302.0
31       3       0 2024-01-02  301.0            302.0            303.0
32       3       0 2024-01-03  302.0            303.0            304.0
33       3       0 2024-01-04  303.0            304.0            305.0
34       3       0 2024-01-05  304.0            305.0            306.0
35       3       0 2024-01-06  305.0            306.0            307.0
36       3       0 2024-01-07  306.0            307.0            308.0
37       3       0 2024-01-08  307.0            308.0            309.0

Encoding mapping:
  symbol:
    AAPL -> 0
    AMZN -> 1
    GOOGL -> 2
    MSFT -> 3
  sector:
    Consumer -> 0
    Tech -> 1

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
   Categorical columns (2): ['symbol', 'sector']
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
   Scaling features for 4 groups
   Groups: [(0, 1), (1, 0), (2, 1), (3, 0)]
   Group (0, 1): fitted scaler on 8 samples
   Group (1, 0): fitted scaler on 8 samples
   Group (2, 1): fitted scaler on 8 samples
   Group (3, 0): fitted scaler on 8 samples

📊 Scaler Parameters (per group):

   Group 0 (AAPL):

   Group 1 (AMZN):

   Group 2 (GOOGL):

   Group 3 (MSFT):

================================================================================
📊 After _scale_features_grouped() [SCALED]
================================================================================
Total rows: 32
Columns (18): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']
    symbol  sector       date     close  close_target_h1  close_target_h2
0        0       1 2024-01-01 -1.527525        -1.527525        -1.527525
1        0       1 2024-01-02 -1.091089        -1.091089        -1.091089
2        0       1 2024-01-03 -0.654654        -0.654654        -0.654654
3        0       1 2024-01-04 -0.218218        -0.218218        -0.218218
4        0       1 2024-01-05  0.218218         0.218218         0.218218
5        0       1 2024-01-06  0.654654         0.654654         0.654654
6        0       1 2024-01-07  1.091089         1.091089         1.091089
7        0       1 2024-01-08  1.527525         1.527525         1.527525
10       1       0 2024-01-01 -1.527525        -1.527525        -1.527525
11       1       0 2024-01-02 -1.091089        -1.091089        -1.091089
12       1       0 2024-01-03 -0.654654        -0.654654        -0.654654
13       1       0 2024-01-04 -0.218218        -0.218218        -0.218218
14       1       0 2024-01-05  0.218218         0.218218         0.218218
15       1       0 2024-01-06  0.654654         0.654654         0.654654
16       1       0 2024-01-07  1.091089         1.091089         1.091089
17       1       0 2024-01-08  1.527525         1.527525         1.527525
20       2       1 2024-01-01 -1.527525        -1.527525        -1.527525
21       2       1 2024-01-02 -1.091089        -1.091089        -1.091089
22       2       1 2024-01-03 -0.654654        -0.654654        -0.654654
23       2       1 2024-01-04 -0.218218        -0.218218        -0.218218
24       2       1 2024-01-05  0.218218         0.218218         0.218218
25       2       1 2024-01-06  0.654654         0.654654         0.654654
26       2       1 2024-01-07  1.091089         1.091089         1.091089
27       2       1 2024-01-08  1.527525         1.527525         1.527525
30       3       0 2024-01-01 -1.527525        -1.527525        -1.527525
31       3       0 2024-01-02 -1.091089        -1.091089        -1.091089
32       3       0 2024-01-03 -0.654654        -0.654654        -0.654654
33       3       0 2024-01-04 -0.218218        -0.218218        -0.218218
34       3       0 2024-01-05  0.218218         0.218218         0.218218
35       3       0 2024-01-06  0.654654         0.654654         0.654654
36       3       0 2024-01-07  1.091089         1.091089         1.091089
37       3       0 2024-01-08  1.527525         1.527525         1.527525

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
  - Total: 6 × 4 groups = 24 sequences (MULTI-COLUMN GROUPING)
  Created 24 sequences from 4 groups
  X_num shape: torch.Size([24, 3, 12]), X_cat shape: torch.Size([24, 2]), y shape: torch.Size([24, 2])

✅ Sequences created:
   X_num shape: torch.Size([24, 3, 12])
   X_cat shape: torch.Size([24, 2])
   y shape: torch.Size([24, 2])

   Expected: 24 sequences total (6 per group × 4 groups)
   Got: 24 sequences

   Group indices: [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (3, 0), (3, 0), (3, 0), (3, 0), (3, 0), (3, 0)]

================================================================================
STEP 7: Training model and running ACTUAL evaluation
================================================================================

🔧 Training a simple model for 5 epochs...
   Sorted data by symbol + sector and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
   Created multi-horizon targets for: close
   Prediction horizons: 1 to 2 steps ahead
   Group-based shifting applied using column: ['symbol', 'sector']
   Remaining samples after shift: 32
   Encoded 'symbol': 4 unique categories
   Encoded 'sector': 2 unique categories
   Categorical cardinalities: {'symbol': 4, 'sector': 2}

   Target columns INCLUDED in sequence: ['close']
   Model will have autoregressive information (use_lagged_target_features=True)
   Excluding non-numeric column: date (dtype: datetime64[ns])
   Excluding non-numeric column: timestamp (dtype: datetime64[ns])
   Scaling features for 4 groups
   Groups: [(0, 1), (1, 0), (2, 1), (3, 0)]
   Group (0, 1): fitted scaler on 8 samples
   Group (1, 0): fitted scaler on 8 samples
   Group (2, 1): fitted scaler on 8 samples
   Group (3, 0): fitted scaler on 8 samples
  Created 24 sequences from 4 groups
  X_num shape: torch.Size([24, 3, 12]), X_cat shape: torch.Size([24, 2]), y shape: torch.Size([24, 2])
⚠️  Training failed: FTTransformerCLSModel.__init__() got an unexpected keyword argument 'd_token'
   Continuing without trained model...

================================================================================
STEP 8: Running ACTUAL evaluation to verify alignment
================================================================================

📋 Testing _evaluate_per_group() method...
   This will show exactly what the predictor extracts for actuals


❌ Evaluation failed: Model must be trained first. Call fit().

🔍 Let's manually inspect what would be extracted...

================================================================================
STEP 9: Manual verification of shifted column extraction
================================================================================

🔮 Making predictions to populate internal state...
⚠️  Prediction failed: Model must be trained first. Call fit().
   Using df_step2 as _last_processed_df

📦 Inspecting _last_processed_df (used for evaluation):
   Total rows: 32
   Per group: {'Consumer': 16, 'Tech': 16}
   Columns: ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2']

   Shifted target columns: ['close_target_h1', 'close_target_h2']

   Extraction offset: 2 (sequence_length - 1)

❌ TEST 1 FAILED: (0, 1)


================================================================================
🔍 TEST 2: MULTI-TARGET ALIGNMENT (MULTI-COLUMN GROUPING)
================================================================================

Configuration:
  - Targets: close, volume (MULTI-TARGET)
  - Groups: 4 with MULTI-COLUMN grouping ['symbol', 'sector']
    * (AAPL, Tech)
    * (GOOGL, Tech)
    * (MSFT, Consumer)
    * (AMZN, Consumer)
  - Rows per group: 10
  - Sequence length: 3
  - Prediction horizon: 2

================================================================================
Creating Multi-Target TimeSeriesPredictor with MULTI-COLUMN grouping...
================================================================================

✅ Multi-target predictor created
   Target columns: ['close', 'volume']
   Group columns: ['symbol', 'sector']
   Is multi-target: True

--- Running MULTI-TARGET-MULTIGROUP test ---

================================================================================
STEP 1: _create_base_features()
================================================================================
This step: sorts by group+time, adds date features

💡 NOTES:
   - Synthetic data has 'timestamp' column (copy of 'date' for testing)
   - Sorting uses group_columns: ['symbol', 'sector']
   - Data is sorted by (group_key, timestamp) for proper temporal order
   - Cyclical encoding: date features → sin/cos pairs (month_sin, month_cos, etc.)
   Sorted data by symbol + sector and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']

================================================================================
📊 After _create_base_features()
================================================================================
Total rows: 40
Columns (16): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
         date   open   high    low  close  volume  month_sin  month_cos
0  2024-01-01   99.0  101.0   99.0  100.0    1000        0.5   0.866025
1  2024-01-02  100.0  102.0  100.0  101.0    1010        0.5   0.866025
2  2024-01-03  101.0  103.0  101.0  102.0    1020        0.5   0.866025
3  2024-01-04  102.0  104.0  102.0  103.0    1030        0.5   0.866025
4  2024-01-05  103.0  105.0  103.0  104.0    1040        0.5   0.866025
5  2024-01-06  104.0  106.0  104.0  105.0    1050        0.5   0.866025
6  2024-01-07  105.0  107.0  105.0  106.0    1060        0.5   0.866025
7  2024-01-08  106.0  108.0  106.0  107.0    1070        0.5   0.866025
8  2024-01-09  107.0  109.0  107.0  108.0    1080        0.5   0.866025
9  2024-01-10  108.0  110.0  108.0  109.0    1090        0.5   0.866025
10 2024-01-01  399.0  401.0  399.0  400.0    4000        0.5   0.866025
11 2024-01-02  400.0  402.0  400.0  401.0    4040        0.5   0.866025
12 2024-01-03  401.0  403.0  401.0  402.0    4080        0.5   0.866025
13 2024-01-04  402.0  404.0  402.0  403.0    4120        0.5   0.866025
14 2024-01-05  403.0  405.0  403.0  404.0    4160        0.5   0.866025
15 2024-01-06  404.0  406.0  404.0  405.0    4200        0.5   0.866025
16 2024-01-07  405.0  407.0  405.0  406.0    4240        0.5   0.866025
17 2024-01-08  406.0  408.0  406.0  407.0    4280        0.5   0.866025
18 2024-01-09  407.0  409.0  407.0  408.0    4320        0.5   0.866025
19 2024-01-10  408.0  410.0  408.0  409.0    4360        0.5   0.866025
20 2024-01-01  199.0  201.0  199.0  200.0    2000        0.5   0.866025
21 2024-01-02  200.0  202.0  200.0  201.0    2020        0.5   0.866025
22 2024-01-03  201.0  203.0  201.0  202.0    2040        0.5   0.866025
23 2024-01-04  202.0  204.0  202.0  203.0    2060        0.5   0.866025
24 2024-01-05  203.0  205.0  203.0  204.0    2080        0.5   0.866025
25 2024-01-06  204.0  206.0  204.0  205.0    2100        0.5   0.866025
26 2024-01-07  205.0  207.0  205.0  206.0    2120        0.5   0.866025
27 2024-01-08  206.0  208.0  206.0  207.0    2140        0.5   0.866025
28 2024-01-09  207.0  209.0  207.0  208.0    2160        0.5   0.866025
29 2024-01-10  208.0  210.0  208.0  209.0    2180        0.5   0.866025
30 2024-01-01  299.0  301.0  299.0  300.0    3000        0.5   0.866025
31 2024-01-02  300.0  302.0  300.0  301.0    3030        0.5   0.866025
32 2024-01-03  301.0  303.0  301.0  302.0    3060        0.5   0.866025
33 2024-01-04  302.0  304.0  302.0  303.0    3090        0.5   0.866025
34 2024-01-05  303.0  305.0  303.0  304.0    3120        0.5   0.866025
35 2024-01-06  304.0  306.0  304.0  305.0    3150        0.5   0.866025
36 2024-01-07  305.0  307.0  305.0  306.0    3180        0.5   0.866025
37 2024-01-08  306.0  308.0  306.0  307.0    3210        0.5   0.866025
38 2024-01-09  307.0  309.0  307.0  308.0    3240        0.5   0.866025
39 2024-01-10  308.0  310.0  308.0  309.0    3270        0.5   0.866025

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
   Group-based shifting applied using column: ['symbol', 'sector']
   Remaining samples after shift: 32

================================================================================
📊 After create_shifted_targets()
================================================================================
Total rows: 32
Columns (20): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

--- symbol=AAPL + sector=Tech (8 rows) ---
  symbol sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
0   AAPL   Tech 2024-01-01  100.0            101.0            102.0    1000            1010.0            1020.0
1   AAPL   Tech 2024-01-02  101.0            102.0            103.0    1010            1020.0            1030.0
2   AAPL   Tech 2024-01-03  102.0            103.0            104.0    1020            1030.0            1040.0
3   AAPL   Tech 2024-01-04  103.0            104.0            105.0    1030            1040.0            1050.0
4   AAPL   Tech 2024-01-05  104.0            105.0            106.0    1040            1050.0            1060.0
5   AAPL   Tech 2024-01-06  105.0            106.0            107.0    1050            1060.0            1070.0
6   AAPL   Tech 2024-01-07  106.0            107.0            108.0    1060            1070.0            1080.0
7   AAPL   Tech 2024-01-08  107.0            108.0            109.0    1070            1080.0            1090.0

--- symbol=AMZN + sector=Consumer (8 rows) ---
   symbol    sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
10   AMZN  Consumer 2024-01-01  400.0            401.0            402.0    4000            4040.0            4080.0
11   AMZN  Consumer 2024-01-02  401.0            402.0            403.0    4040            4080.0            4120.0
12   AMZN  Consumer 2024-01-03  402.0            403.0            404.0    4080            4120.0            4160.0
13   AMZN  Consumer 2024-01-04  403.0            404.0            405.0    4120            4160.0            4200.0
14   AMZN  Consumer 2024-01-05  404.0            405.0            406.0    4160            4200.0            4240.0
15   AMZN  Consumer 2024-01-06  405.0            406.0            407.0    4200            4240.0            4280.0
16   AMZN  Consumer 2024-01-07  406.0            407.0            408.0    4240            4280.0            4320.0
17   AMZN  Consumer 2024-01-08  407.0            408.0            409.0    4280            4320.0            4360.0

--- symbol=GOOGL + sector=Tech (8 rows) ---
   symbol sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
20  GOOGL   Tech 2024-01-01  200.0            201.0            202.0    2000            2020.0            2040.0
21  GOOGL   Tech 2024-01-02  201.0            202.0            203.0    2020            2040.0            2060.0
22  GOOGL   Tech 2024-01-03  202.0            203.0            204.0    2040            2060.0            2080.0
23  GOOGL   Tech 2024-01-04  203.0            204.0            205.0    2060            2080.0            2100.0
24  GOOGL   Tech 2024-01-05  204.0            205.0            206.0    2080            2100.0            2120.0
25  GOOGL   Tech 2024-01-06  205.0            206.0            207.0    2100            2120.0            2140.0
26  GOOGL   Tech 2024-01-07  206.0            207.0            208.0    2120            2140.0            2160.0
27  GOOGL   Tech 2024-01-08  207.0            208.0            209.0    2140            2160.0            2180.0

--- symbol=MSFT + sector=Consumer (8 rows) ---
   symbol    sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
30   MSFT  Consumer 2024-01-01  300.0            301.0            302.0    3000            3030.0            3060.0
31   MSFT  Consumer 2024-01-02  301.0            302.0            303.0    3030            3060.0            3090.0
32   MSFT  Consumer 2024-01-03  302.0            303.0            304.0    3060            3090.0            3120.0
33   MSFT  Consumer 2024-01-04  303.0            304.0            305.0    3090            3120.0            3150.0
34   MSFT  Consumer 2024-01-05  304.0            305.0            306.0    3120            3150.0            3180.0
35   MSFT  Consumer 2024-01-06  305.0            306.0            307.0    3150            3180.0            3210.0
36   MSFT  Consumer 2024-01-07  306.0            307.0            308.0    3180            3210.0            3240.0
37   MSFT  Consumer 2024-01-08  307.0            308.0            309.0    3210            3240.0            3270.0

🔑 KEY POINT: This dataframe (df_step2) is what gets stored in _last_processed_df
   It's BEFORE encoding and scaling, so ['symbol', 'sector'] are still strings/original values
   Row count per group: {('AAPL', 'Tech'): 8, ('AMZN', 'Consumer'): 8, ('GOOGL', 'Tech'): 8, ('MSFT', 'Consumer'): 8}

================================================================================
STEP 3: _encode_categorical_features()
================================================================================
This step: encodes 'symbol' as integers (AAPL=0, GOOGL=1)
   Encoded 'symbol': 4 unique categories
   Encoded 'sector': 2 unique categories
   Categorical cardinalities: {'symbol': 4, 'sector': 2}

================================================================================
📊 After _encode_categorical_features()
================================================================================
Total rows: 32
Columns (20): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

--- symbol=0 + sector=1 (8 rows) ---
   symbol  sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
0       0       1 2024-01-01  100.0            101.0            102.0    1000            1010.0            1020.0
1       0       1 2024-01-02  101.0            102.0            103.0    1010            1020.0            1030.0
2       0       1 2024-01-03  102.0            103.0            104.0    1020            1030.0            1040.0
3       0       1 2024-01-04  103.0            104.0            105.0    1030            1040.0            1050.0
4       0       1 2024-01-05  104.0            105.0            106.0    1040            1050.0            1060.0
5       0       1 2024-01-06  105.0            106.0            107.0    1050            1060.0            1070.0
6       0       1 2024-01-07  106.0            107.0            108.0    1060            1070.0            1080.0
7       0       1 2024-01-08  107.0            108.0            109.0    1070            1080.0            1090.0

--- symbol=1 + sector=0 (8 rows) ---
    symbol  sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
10       1       0 2024-01-01  400.0            401.0            402.0    4000            4040.0            4080.0
11       1       0 2024-01-02  401.0            402.0            403.0    4040            4080.0            4120.0
12       1       0 2024-01-03  402.0            403.0            404.0    4080            4120.0            4160.0
13       1       0 2024-01-04  403.0            404.0            405.0    4120            4160.0            4200.0
14       1       0 2024-01-05  404.0            405.0            406.0    4160            4200.0            4240.0
15       1       0 2024-01-06  405.0            406.0            407.0    4200            4240.0            4280.0
16       1       0 2024-01-07  406.0            407.0            408.0    4240            4280.0            4320.0
17       1       0 2024-01-08  407.0            408.0            409.0    4280            4320.0            4360.0

--- symbol=2 + sector=1 (8 rows) ---
    symbol  sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
20       2       1 2024-01-01  200.0            201.0            202.0    2000            2020.0            2040.0
21       2       1 2024-01-02  201.0            202.0            203.0    2020            2040.0            2060.0
22       2       1 2024-01-03  202.0            203.0            204.0    2040            2060.0            2080.0
23       2       1 2024-01-04  203.0            204.0            205.0    2060            2080.0            2100.0
24       2       1 2024-01-05  204.0            205.0            206.0    2080            2100.0            2120.0
25       2       1 2024-01-06  205.0            206.0            207.0    2100            2120.0            2140.0
26       2       1 2024-01-07  206.0            207.0            208.0    2120            2140.0            2160.0
27       2       1 2024-01-08  207.0            208.0            209.0    2140            2160.0            2180.0

--- symbol=3 + sector=0 (8 rows) ---
    symbol  sector       date  close  close_target_h1  close_target_h2  volume  volume_target_h1  volume_target_h2
30       3       0 2024-01-01  300.0            301.0            302.0    3000            3030.0            3060.0
31       3       0 2024-01-02  301.0            302.0            303.0    3030            3060.0            3090.0
32       3       0 2024-01-03  302.0            303.0            304.0    3060            3090.0            3120.0
33       3       0 2024-01-04  303.0            304.0            305.0    3090            3120.0            3150.0
34       3       0 2024-01-05  304.0            305.0            306.0    3120            3150.0            3180.0
35       3       0 2024-01-06  305.0            306.0            307.0    3150            3180.0            3210.0
36       3       0 2024-01-07  306.0            307.0            308.0    3180            3210.0            3240.0
37       3       0 2024-01-08  307.0            308.0            309.0    3210            3240.0            3270.0

Encoding mapping:
  symbol:
    AAPL -> 0
    AMZN -> 1
    GOOGL -> 2
    MSFT -> 3
  sector:
    Consumer -> 0
    Tech -> 1

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
   Categorical columns (2): ['symbol', 'sector']
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
   Scaling features for 4 groups
   Groups: [(0, 1), (1, 0), (2, 1), (3, 0)]
   Group (0, 1): fitted scaler on 8 samples
   Group (1, 0): fitted scaler on 8 samples
   Group (2, 1): fitted scaler on 8 samples
   Group (3, 0): fitted scaler on 8 samples

📊 Scaler Parameters (per group):

   Group 0 (AAPL):

   Group 1 (AMZN):

   Group 2 (GOOGL):

   Group 3 (MSFT):

================================================================================
📊 After _scale_features_grouped() [SCALED]
================================================================================
Total rows: 32
Columns (20): ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']
    symbol  sector       date     close  close_target_h1  close_target_h2    volume  volume_target_h1  volume_target_h2
0        0       1 2024-01-01 -1.527525        -1.527525        -1.527525 -1.527525         -1.527525         -1.527525
1        0       1 2024-01-02 -1.091089        -1.091089        -1.091089 -1.091089         -1.091089         -1.091089
2        0       1 2024-01-03 -0.654654        -0.654654        -0.654654 -0.654654         -0.654654         -0.654654
3        0       1 2024-01-04 -0.218218        -0.218218        -0.218218 -0.218218         -0.218218         -0.218218
4        0       1 2024-01-05  0.218218         0.218218         0.218218  0.218218          0.218218          0.218218
5        0       1 2024-01-06  0.654654         0.654654         0.654654  0.654654          0.654654          0.654654
6        0       1 2024-01-07  1.091089         1.091089         1.091089  1.091089          1.091089          1.091089
7        0       1 2024-01-08  1.527525         1.527525         1.527525  1.527525          1.527525          1.527525
10       1       0 2024-01-01 -1.527525        -1.527525        -1.527525 -1.527525         -1.527525         -1.527525
11       1       0 2024-01-02 -1.091089        -1.091089        -1.091089 -1.091089         -1.091089         -1.091089
12       1       0 2024-01-03 -0.654654        -0.654654        -0.654654 -0.654654         -0.654654         -0.654654
13       1       0 2024-01-04 -0.218218        -0.218218        -0.218218 -0.218218         -0.218218         -0.218218
14       1       0 2024-01-05  0.218218         0.218218         0.218218  0.218218          0.218218          0.218218
15       1       0 2024-01-06  0.654654         0.654654         0.654654  0.654654          0.654654          0.654654
16       1       0 2024-01-07  1.091089         1.091089         1.091089  1.091089          1.091089          1.091089
17       1       0 2024-01-08  1.527525         1.527525         1.527525  1.527525          1.527525          1.527525
20       2       1 2024-01-01 -1.527525        -1.527525        -1.527525 -1.527525         -1.527525         -1.527525
21       2       1 2024-01-02 -1.091089        -1.091089        -1.091089 -1.091089         -1.091089         -1.091089
22       2       1 2024-01-03 -0.654654        -0.654654        -0.654654 -0.654654         -0.654654         -0.654654
23       2       1 2024-01-04 -0.218218        -0.218218        -0.218218 -0.218218         -0.218218         -0.218218
24       2       1 2024-01-05  0.218218         0.218218         0.218218  0.218218          0.218218          0.218218
25       2       1 2024-01-06  0.654654         0.654654         0.654654  0.654654          0.654654          0.654654
26       2       1 2024-01-07  1.091089         1.091089         1.091089  1.091089          1.091089          1.091089
27       2       1 2024-01-08  1.527525         1.527525         1.527525  1.527525          1.527525          1.527525
30       3       0 2024-01-01 -1.527525        -1.527525        -1.527525 -1.527525         -1.527525         -1.527525
31       3       0 2024-01-02 -1.091089        -1.091089        -1.091089 -1.091089         -1.091089         -1.091089
32       3       0 2024-01-03 -0.654654        -0.654654        -0.654654 -0.654654         -0.654654         -0.654654
33       3       0 2024-01-04 -0.218218        -0.218218        -0.218218 -0.218218         -0.218218         -0.218218
34       3       0 2024-01-05  0.218218         0.218218         0.218218  0.218218          0.218218          0.218218
35       3       0 2024-01-06  0.654654         0.654654         0.654654  0.654654          0.654654          0.654654
36       3       0 2024-01-07  1.091089         1.091089         1.091089  1.091089          1.091089          1.091089
37       3       0 2024-01-08  1.527525         1.527525         1.527525  1.527525          1.527525          1.527525

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
  - Total: 6 × 4 groups = 24 sequences (MULTI-COLUMN GROUPING)
  Created 24 sequences from 4 groups
  X_num shape: torch.Size([24, 3, 12]), X_cat shape: torch.Size([24, 2]), y shape: torch.Size([24, 4])

✅ Sequences created:
   X_num shape: torch.Size([24, 3, 12])
   X_cat shape: torch.Size([24, 2])
   y shape: torch.Size([24, 4])

   Expected: 24 sequences total (6 per group × 4 groups)
   Got: 24 sequences

   Group indices: [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (3, 0), (3, 0), (3, 0), (3, 0), (3, 0), (3, 0)]

================================================================================
STEP 7: Training model and running ACTUAL evaluation
================================================================================

🔧 Training a simple model for 5 epochs...
   Sorted data by symbol + sector and 'timestamp' to ensure temporal order within groups
   Data already chronologically sorted within groups by 'timestamp'
   Dropped non-cyclical temporal features: ['month', 'day', 'dayofweek']
   Kept cyclical features: ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos']
   Created multi-horizon targets for: close, volume
   Prediction horizons: 1 to 2 steps ahead
   Group-based shifting applied using column: ['symbol', 'sector']
   Remaining samples after shift: 32
   Encoded 'symbol': 4 unique categories
   Encoded 'sector': 2 unique categories
   Categorical cardinalities: {'symbol': 4, 'sector': 2}

   Target columns INCLUDED in sequence: ['close', 'volume']
   Model will have autoregressive information (use_lagged_target_features=True)
   Excluding non-numeric column: date (dtype: datetime64[ns])
   Excluding non-numeric column: timestamp (dtype: datetime64[ns])
   Scaling features for 4 groups
   Groups: [(0, 1), (1, 0), (2, 1), (3, 0)]
   Group (0, 1): fitted scaler on 8 samples
   Group (1, 0): fitted scaler on 8 samples
   Group (2, 1): fitted scaler on 8 samples
   Group (3, 0): fitted scaler on 8 samples
  Created 24 sequences from 4 groups
  X_num shape: torch.Size([24, 3, 12]), X_cat shape: torch.Size([24, 2]), y shape: torch.Size([24, 4])
⚠️  Training failed: FTTransformerCLSModel.__init__() got an unexpected keyword argument 'd_token'
   Continuing without trained model...

================================================================================
STEP 8: Running ACTUAL evaluation to verify alignment
================================================================================

📋 Testing _evaluate_per_group() method...
   This will show exactly what the predictor extracts for actuals


❌ Evaluation failed: Model must be trained first. Call fit().

🔍 Let's manually inspect what would be extracted...

================================================================================
STEP 9: Manual verification of shifted column extraction
================================================================================

🔮 Making predictions to populate internal state...
⚠️  Prediction failed: Model must be trained first. Call fit().
   Using df_step2 as _last_processed_df

📦 Inspecting _last_processed_df (used for evaluation):
   Total rows: 32
   Per group: {'Consumer': 16, 'Tech': 16}
   Columns: ['symbol', 'sector', 'date', 'close', 'volume', 'open', 'high', 'low', 'timestamp', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'day_sin', 'day_cos', 'close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

   Shifted target columns: ['close_target_h1', 'close_target_h2', 'volume_target_h1', 'volume_target_h2']

   Extraction offset: 2 (sequence_length - 1)

❌ TEST 2 FAILED: (0, 1)


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
