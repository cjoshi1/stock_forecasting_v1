# Parameter Naming Issue Analysis and Fix Plan

**Date:** 2025-11-06
**Issue:** `TransformerBasedModel.__init__() got an unexpected keyword argument 'd_token'`
**Status:** Identified root cause, fix plan ready

---

## Root Cause Analysis

### What Happened

We did a **partial standardization** of parameter names from:
- OLD: `d_model`, `num_heads`, `num_layers`
- NEW: `d_token`, `n_heads`, `n_layers`

### Current State of Each Class

```
Class Hierarchy and Parameter Names:
════════════════════════════════════════════════════════════════

TransformerBasedModel (base class)           ❌ NEEDS FIX
├─ Line 117: __init__(d_model, num_heads, num_layers)  ← ALL OLD
├─ Line 127: self.d_model = d_model
├─ Line 128: self.num_heads = num_heads
├─ Line 129: self.num_layers = num_layers
└─ Line 133: return self.d_model  (in get_embedding_dim)

FTTransformer (base, non-CLS)                ✅ ALREADY UPDATED
├─ Line 451: d_token: int = 192              ← NEW
├─ Line 452: n_layers: int = 3               ← NEW
├─ Line 453: n_heads: int = 8                ← NEW
└─ Does NOT inherit from TransformerBasedModel

FTTransformerCLSModel                        ✅ ALREADY UPDATED
├─ Line 650: Inherits from TransformerBasedModel
├─ Line 683: __init__(..., d_token, n_heads, n_layers)  ← NEW
├─ Line 706: super().__init__(d_token=d_token, n_heads=n_heads, n_layers=n_layers)
│                              ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
│                        ❌ ERROR: Parent expects OLD names!
└─ Line 710: self.d_token = d_token

CSNTransformer (base, non-CLS)               ⚠️ INCONSISTENT
├─ Line 626: d_model: int = 128              ← OLD (should be d_token)
├─ Line 627: n_layers: int = 3               ← NEW
├─ Line 628: n_heads: int = 8                ← NEW
├─ Line 634: self.d_model = d_model          ← Uses old name
└─ Does NOT inherit from TransformerBasedModel

CSNTransformerCLSModel                       ✅ ALREADY UPDATED
├─ Line 813: Inherits from TransformerBasedModel
├─ Line 870: __init__(..., d_token, n_heads, n_layers)  ← NEW
├─ Line 905: super().__init__(d_token=d_token, n_heads=n_heads, n_layers=n_layers)
│                              ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
│                        ❌ ERROR: Parent expects OLD names!
└─ Line 909: self.d_token = d_token
```

### The Error Chain

```
1. User runs: debug_alignment_simple.py
   ↓
2. Script creates: TimeSeriesPredictor(d_token=32, n_heads=2, n_layers=2)
   ↓
3. Predictor creates: FTTransformerCLSModel(d_token=32, n_heads=2, n_layers=2)
   ↓
4. FTTransformerCLSModel.__init__ calls:
   super().__init__(d_token=32, n_heads=2, n_layers=2)
   ↓
5. This calls: TransformerBasedModel.__init__(d_token=32, n_heads=2, n_layers=2)
   ↓
6. ❌ ERROR: TransformerBasedModel.__init__ signature is:
   __init__(self, d_model, num_heads, num_layers)

   It doesn't recognize keyword arguments:
   - d_token (expects d_model)
   - n_heads (expects num_heads)
   - n_layers (expects num_layers)
```

### Why This Happened

When we standardized parameters, we updated:
- ✅ Child classes (FTTransformerCLSModel, CSNTransformerCLSModel)
- ❌ Parent class (TransformerBasedModel) - **FORGOT THIS!**
- ⚠️ CSNTransformer - Only partially updated

---

## Fix Plan

### Step 1: Update TransformerBasedModel Base Class

**File:** `tf_predictor/core/base/model_interface.py`

**Changes needed:**

```python
# BEFORE (lines 117-133):
class TransformerBasedModel(TimeSeriesModel):
    def __init__(self, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension (d_model)."""
        return self.d_model

# AFTER:
class TransformerBasedModel(TimeSeriesModel):
    def __init__(self, d_token: int, n_heads: int, n_layers: int):
        """
        Initialize transformer-based model.

        Args:
            d_token: Token embedding dimension (formerly d_model)
            n_heads: Number of attention heads (formerly num_heads)
            n_layers: Number of transformer layers (formerly num_layers)
        """
        super().__init__()
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers

    def get_embedding_dim(self) -> int:
        """Get the token embedding dimension (d_token)."""
        return self.d_token
```

**Impact:**
- Affects: All classes inheriting from TransformerBasedModel
- Benefits: Consistent naming across entire codebase

---

### Step 2: Update CSNTransformer Base Class

**File:** `tf_predictor/core/csn_model.py`

**Changes needed:**

Line 626: Change `d_model` to `d_token`:
```python
# BEFORE:
def __init__(self,
             categorical_features: Dict[str, int],
             num_numerical_features: int,
             sequence_length: int,
             d_model: int = 128,
             n_layers: int = 3,
             n_heads: int = 8,
             ...

# AFTER:
def __init__(self,
             categorical_features: Dict[str, int],
             num_numerical_features: int,
             sequence_length: int,
             d_token: int = 128,
             n_layers: int = 3,
             n_heads: int = 8,
             ...
```

Line 634: Update instance variable:
```python
# BEFORE:
self.d_model = d_model

# AFTER:
self.d_token = d_token
```

**Search for all references to `self.d_model` in csn_model.py and update to `self.d_token`**

---

### Step 3: Update Documentation Comments

Update example configs in docstrings that still reference old names:

**File:** `tf_predictor/core/base/model_interface.py`

Line 60-69 example config:
```python
# BEFORE:
Example:
    {
        'model_type': 'ft_transformer',
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 3,
        ...
    }

# AFTER:
Example:
    {
        'model_type': 'ft_transformer',
        'd_token': 64,
        'n_heads': 4,
        'n_layers': 3,
        ...
    }
```

---

## Verification Steps

After making the changes, verify:

### 1. Check All super().__init__() Calls Match

```bash
# Should find FTTransformerCLSModel and CSNTransformerCLSModel calling:
# super().__init__(d_token=..., n_heads=..., n_layers=...)
grep -n "super().__init__" tf_predictor/core/ft_model.py tf_predictor/core/csn_model.py
```

### 2. Verify No Remaining d_model References in CSNTransformer

```bash
# Should only find references in comments/docstrings, not code
grep -n "d_model" tf_predictor/core/csn_model.py
```

### 3. Run the Alignment Test

```bash
PYTHONPATH=. python debug_alignment_simple.py
```

Expected: Should run without parameter name errors

---

## Files to Modify

| File | Lines | Changes |
|------|-------|---------|
| `tf_predictor/core/base/model_interface.py` | 117-133 | Update TransformerBasedModel.__init__ signature and instance vars |
| `tf_predictor/core/base/model_interface.py` | 60-69 | Update example config in docstring |
| `tf_predictor/core/csn_model.py` | 626 | Change d_model parameter to d_token |
| `tf_predictor/core/csn_model.py` | 634 | Change self.d_model to self.d_token |
| `tf_predictor/core/csn_model.py` | All | Update all references to self.d_model → self.d_token |

---

## Risk Assessment

**Low Risk:**
- TransformerBasedModel is only used by our CLS models
- Both CLS models already expect the new names
- No external code depends on this internal base class

**Breaking Changes:**
- None externally (this is internal refactoring)
- Existing saved models should still load (they save d_token in config)

**Testing Needed:**
1. Run debug_alignment_simple.py (immediate verification)
2. Run existing test suite if available
3. Verify model loading/saving still works

---

## Summary

**Problem:** Incomplete parameter standardization left base class with old names while child classes use new names.

**Solution:** Complete the standardization by updating:
1. TransformerBasedModel base class (model_interface.py)
2. CSNTransformer base class (csn_model.py)
3. Documentation examples

**Estimated Time:** 15-20 minutes
**Estimated Risk:** Low (internal refactoring only)

---

**Ready to proceed with fixes?** All changes are straightforward search-and-replace operations.
