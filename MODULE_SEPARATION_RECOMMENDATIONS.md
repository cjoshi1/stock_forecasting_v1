# Module Separation Recommendations

**Date**: 2025-10-30
**Status**: 📋 RECOMMENDATIONS

## Current Architecture Analysis

### ✅ What's Already Good

Your current architecture already follows many good practices:

1. **Clean Dependency Direction** ✅
   - `tf_predictor` has NO imports from `daily_stock_forecasting` or `intraday_forecasting`
   - Both applications depend on `tf_predictor`, not the other way around
   - This is exactly the right dependency direction!

2. **Proper Inheritance Model** ✅
   - `StockPredictor(TimeSeriesPredictor)` - extends base with stock-specific features
   - `IntradayPredictor(TimeSeriesPredictor)` - extends base with intraday-specific features
   - Domain logic stays in domain modules

3. **Extension Pattern via `prepare_features()`** ✅
   - Applications override `prepare_features()` to add domain-specific features
   - Base class provides time-series functionality
   - Clean separation of concerns

### Current Structure

```
stock_forecasting_v1/
├── tf_predictor/                      # Generic time-series module
│   ├── core/
│   │   ├── predictor.py               # TimeSeriesPredictor (base class)
│   │   ├── ft_model.py                # FT-Transformer model
│   │   ├── csn_model.py               # CSN-Transformer model
│   │   └── base/
│   │       ├── model_factory.py       # Model factory
│   │       └── model_interface.py     # Model interface
│   ├── preprocessing/
│   │   ├── time_features.py           # Generic time-series features
│   │   └── scaler_factory.py          # Scaler factory
│   └── __init__.py                    # Clean exports
│
├── daily_stock_forecasting/           # Stock application
│   ├── predictor.py                   # StockPredictor (extends TimeSeriesPredictor)
│   ├── preprocessing/
│   │   └── stock_features.py          # Stock-specific features (technical indicators)
│   └── visualization/
│       └── stock_charts.py            # Stock-specific visualizations
│
└── intraday_forecasting/              # Intraday application
    ├── predictor.py                   # IntradayPredictor (extends TimeSeriesPredictor)
    ├── preprocessing/
    │   ├── intraday_features.py       # Intraday-specific features
    │   └── timeframe_utils.py         # Timeframe configurations
    └── visualization/
        └── intraday_charts.py         # Intraday-specific visualizations
```

### Dependency Graph

```
┌─────────────────────────────────────────┐
│         tf_predictor                    │
│  (Generic Time Series Module)           │
│  - No domain dependencies                │
│  - Reusable for any time series         │
└─────────────────────────────────────────┘
                 ▲         ▲
                 │         │
                 │         │
     ┌───────────┘         └────────────┐
     │                                   │
┌────┴──────────────────┐   ┌───────────┴─────────────┐
│ daily_stock_          │   │ intraday_forecasting    │
│ forecasting           │   │                         │
│ (Stock Application)   │   │ (Intraday Application)  │
└───────────────────────┘   └─────────────────────────┘
```

## 🎯 Recommendations for Maintaining Clean Separation

### 1. **Make `tf_predictor` a Standalone Package** (High Priority)

Convert `tf_predictor` into an installable package that can be used independently.

#### Benefits:
- ✅ Use in other projects without copying code
- ✅ Version control and semantic versioning
- ✅ Clear API boundaries
- ✅ Can publish to PyPI (public or private)

#### Implementation:

**Option A: Keep as Local Editable Package**
```bash
# In tf_predictor/ directory, create setup.py:
```

```python
# tf_predictor/setup.py
from setuptools import setup, find_packages

setup(
    name="tf-predictor",
    version="1.0.0",
    description="Generic time series forecasting with FT-Transformer",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)
```

Then install in editable mode:
```bash
cd tf_predictor
pip install -e .

# Now you can use it anywhere:
from tf_predictor import TimeSeriesPredictor
```

**Option B: Use as Git Submodule** (if using across multiple projects)
```bash
# In another project:
git submodule add <your-repo-url>/tf_predictor
pip install -e tf_predictor/
```

### 2. **Create Clear API Boundaries** (High Priority)

Document what's public API vs internal implementation.

#### `tf_predictor/__init__.py` - Public API

```python
"""
Public API for tf_predictor.

Stable interface that applications should depend on.
"""

# Core classes (STABLE API)
from .core.predictor import TimeSeriesPredictor

# Preprocessing utilities (STABLE API)
from .preprocessing.scaler_factory import ScalerFactory, OnlyMaxScaler
from .preprocessing.time_features import (
    create_date_features,
    create_cyclical_features,
    create_shifted_targets,
    create_input_variable_sequence
)

# Utilities (STABLE API)
from .core.utils import calculate_metrics, load_time_series_data, split_time_series

__version__ = "1.0.0"

# Public API - applications should only use these
__all__ = [
    # Core
    "TimeSeriesPredictor",

    # Preprocessing
    "ScalerFactory",
    "OnlyMaxScaler",
    "create_date_features",
    "create_cyclical_features",
    "create_shifted_targets",
    "create_input_variable_sequence",

    # Utilities
    "calculate_metrics",
    "load_time_series_data",
    "split_time_series",
]
```

#### Mark internal modules with underscore prefix

```python
# tf_predictor/core/_internal_utils.py  (private module)
# Applications should NOT import from _internal modules
```

### 3. **Repository Structure Options** (Medium Priority)

Choose one based on your use case:

#### **Option A: Monorepo (Current)** - RECOMMENDED FOR NOW

Keep everything in one repo but with clear boundaries:

```
stock_forecasting_v1/
├── tf_predictor/              # Core library (installable)
│   ├── setup.py               # Makes it pip installable
│   ├── pyproject.toml         # Modern Python packaging
│   └── ...
├── daily_stock_forecasting/   # Application 1
├── intraday_forecasting/      # Application 2
├── requirements.txt           # Application requirements
└── README.md                  # Project overview
```

**Pros**:
- Easy to develop both library and applications together
- Single version control
- Can test library changes with applications immediately

**Cons**:
- Slightly more complex setup
- Need discipline to maintain boundaries

#### **Option B: Separate Repositories** - FOR MATURE LIBRARY

Split into separate repos when `tf_predictor` is stable:

```
Repository 1: tf-predictor/
└── (Just the tf_predictor module as standalone package)

Repository 2: stock-forecasting-apps/
├── daily_stock_forecasting/
├── intraday_forecasting/
└── requirements.txt (includes tf-predictor)
```

**Pros**:
- Physical separation enforces boundaries
- Independent versioning
- Can open-source tf_predictor separately

**Cons**:
- More overhead (2 repos to manage)
- Testing changes requires publish/install cycle

### 4. **Enforce Dependency Rules** (High Priority)

#### Create import linters

**File: `tf_predictor/tests/test_no_domain_imports.py`**

```python
"""
Test to ensure tf_predictor has no dependencies on domain modules.
This keeps the module generic and reusable.
"""

import os
import re
from pathlib import Path

def test_no_domain_imports():
    """Ensure tf_predictor doesn't import from domain modules."""

    # Forbidden imports
    forbidden_patterns = [
        r'from daily_stock_forecasting',
        r'import daily_stock_forecasting',
        r'from intraday_forecasting',
        r'import intraday_forecasting',
        r'from.*stock_features',  # Domain-specific feature files
        r'from.*intraday_features',
    ]

    tf_predictor_dir = Path(__file__).parent.parent
    violations = []

    # Check all Python files in tf_predictor
    for py_file in tf_predictor_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue

        with open(py_file, 'r') as f:
            content = f.read()

        for pattern in forbidden_patterns:
            if re.search(pattern, content):
                violations.append(f"{py_file}: Found forbidden import pattern '{pattern}'")

    if violations:
        error_msg = "\n".join(violations)
        raise AssertionError(
            f"tf_predictor should not import from domain modules:\n{error_msg}"
        )

if __name__ == '__main__':
    test_no_domain_imports()
    print("✓ No domain imports found in tf_predictor")
```

Run this as part of your test suite:
```bash
PYTHONPATH=. python tf_predictor/tests/test_no_domain_imports.py
```

### 5. **Documentation Strategy** (Medium Priority)

#### Separate documentation for each level:

**`tf_predictor/README.md`** - Generic library docs
```markdown
# TF-Predictor: Generic Time Series Forecasting

A reusable time series forecasting library using FT-Transformer.

## Features
- Generic time series prediction
- Multiple model architectures (FT-Transformer, CSN-Transformer)
- Flexible scaler selection
- Multi-target and multi-horizon support
- No domain-specific assumptions

## Usage
[Generic examples with random/synthetic data]

## Extending
[How to create domain-specific predictors]
```

**`daily_stock_forecasting/README.md`** - Stock application docs
```markdown
# Daily Stock Forecasting

Stock market prediction application using TF-Predictor.

## Features
- Technical indicators
- OHLCV data handling
- Stock-specific visualizations

## Usage
[Stock-specific examples]
```

**`intraday_forecasting/README.md`** - Intraday application docs
```markdown
# Intraday Forecasting

High-frequency trading forecasting using TF-Predictor.

## Features
- Minute-level predictions
- Market microstructure features
- Timeframe-specific configurations

## Usage
[Intraday-specific examples]
```

### 6. **Configuration Management** (Low Priority)

Keep configurations separate:

```
tf_predictor/
└── config/
    └── model_defaults.py          # Generic model configs

daily_stock_forecasting/
└── config/
    ├── stock_model_configs.py     # Stock-specific configs
    └── market_calendars.py        # Stock market calendars

intraday_forecasting/
└── config/
    ├── intraday_configs.py        # Intraday-specific configs
    └── timeframe_settings.py      # Timeframe settings
```

### 7. **Testing Strategy** (High Priority)

#### Three levels of tests:

**Level 1: tf_predictor unit tests**
```python
# tf_predictor/tests/test_predictor.py
# Test generic functionality with synthetic data
# NO stock or intraday-specific data
```

**Level 2: Application unit tests**
```python
# daily_stock_forecasting/tests/test_stock_predictor.py
# Test stock-specific features

# intraday_forecasting/tests/test_intraday_predictor.py
# Test intraday-specific features
```

**Level 3: Integration tests**
```python
# Root level: tests/integration/
# Test full pipelines end-to-end
```

### 8. **Version Compatibility** (Medium Priority)

If you make `tf_predictor` installable, applications can pin versions:

```python
# daily_stock_forecasting/requirements.txt
tf-predictor==1.0.0  # Pin to specific version
pandas>=2.0.0
yfinance>=0.2.0

# intraday_forecasting/requirements.txt
tf-predictor>=1.0.0,<2.0.0  # Allow minor updates
pandas>=2.0.0
alpaca-py>=0.5.0
```

### 9. **Namespace Packages** (Advanced, Optional)

For very large projects, consider namespace packages:

```python
# Instead of:
from tf_predictor import TimeSeriesPredictor
from daily_stock_forecasting import StockPredictor

# Use namespace:
from forecasting.core import TimeSeriesPredictor
from forecasting.stock import StockPredictor
from forecasting.intraday import IntradayPredictor
```

But this adds complexity - only do if you have many modules.

## 🎯 Recommended Action Plan

### Phase 1: Documentation & Testing (Week 1)
1. ✅ Add `test_no_domain_imports.py` to enforce boundaries
2. ✅ Update README files for each module
3. ✅ Document public API clearly in `__init__.py`

### Phase 2: Package Setup (Week 2)
1. ✅ Create `tf_predictor/setup.py`
2. ✅ Make tf_predictor pip-installable in editable mode
3. ✅ Update import statements in applications

### Phase 3: Refinement (Week 3)
1. ✅ Add type hints throughout public API
2. ✅ Create comprehensive examples for tf_predictor
3. ✅ Set up CI/CD to run boundary tests

### Phase 4: Publishing (Optional)
1. Consider publishing tf_predictor to private PyPI
2. Or create separate repo for tf_predictor
3. Or keep as monorepo with clear boundaries

## Example: Using tf_predictor for Other Domains

With clean separation, you can easily create new applications:

### Energy Forecasting Application

```python
# energy_forecasting/predictor.py
from tf_predictor import TimeSeriesPredictor
from .preprocessing.energy_features import create_energy_features

class EnergyPredictor(TimeSeriesPredictor):
    """Energy consumption/generation forecasting."""

    def prepare_features(self, df, fit_scaler=False):
        # Add energy-specific features (temperature, season, etc.)
        df_with_energy = create_energy_features(df)
        return super().prepare_features(df_with_energy, fit_scaler)
```

### Weather Forecasting Application

```python
# weather_forecasting/predictor.py
from tf_predictor import TimeSeriesPredictor
from .preprocessing.weather_features import create_weather_features

class WeatherPredictor(TimeSeriesPredictor):
    """Weather forecasting with atmospheric features."""

    def prepare_features(self, df, fit_scaler=False):
        # Add weather-specific features (pressure, humidity, etc.)
        df_with_weather = create_weather_features(df)
        return super().prepare_features(df_with_weather, fit_scaler)
```

## Summary

### ✅ Current Status
Your architecture is already well-designed with:
- Correct dependency direction (applications depend on library, not reverse)
- Clean inheritance model
- Extension points via `prepare_features()`

### 🎯 Key Recommendations
1. **Make tf_predictor pip-installable** (highest impact)
2. **Add import boundary tests** (prevent regression)
3. **Document public API clearly** (help users)
4. **Keep monorepo structure** (works well for your use case)

### 🚀 Next Steps
Start with Phase 1 (documentation and testing), then move to Phase 2 (packaging). This will give you a clean, reusable time series library while keeping development workflow smooth.
