"""
Data loading with caching support for Rossmann forecasting.
"""
import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import hashlib


def load_preprocessing_config(config_name: str = 'baseline') -> Dict[str, Any]:
    """
    Load preprocessing configuration from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    config_path = Path(f'rossman_forecasting/configs/preprocessing/{config_name}.yaml')

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_raw_data(data_dir: str = 'rossman_forecasting/data/raw') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Rossmann data from CSV files.

    Args:
        data_dir: Directory containing raw data files

    Returns:
        Tuple of (train_df, test_df, store_df)
    """
    data_path = Path(data_dir)

    print(f"📂 Loading raw data from {data_dir}...")

    # Load files
    train_df = pd.read_csv(data_path / 'train.csv', parse_dates=['Date'])
    test_df = pd.read_csv(data_path / 'test.csv', parse_dates=['Date'])
    store_df = pd.read_csv(data_path / 'store.csv')

    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    print(f"   Store: {store_df.shape}")

    return train_df, test_df, store_df


def get_cache_path(config: Dict[str, Any], data_type: str) -> Path:
    """
    Get cache file path for processed data.

    Args:
        config: Preprocessing configuration
        data_type: 'train' or 'test'

    Returns:
        Path to cache file
    """
    cache_version = config.get('version', 'unknown')
    cache_dir = Path(f'rossman_forecasting/data/processed/{cache_version}')
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f'{data_type}_processed.csv'


def save_config_with_data(config: Dict[str, Any], cache_dir: Path) -> None:
    """
    Save configuration alongside processed data for reproducibility.

    Args:
        config: Preprocessing configuration
        cache_dir: Directory to save config
    """
    config_path = cache_dir / 'preprocessing_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def check_cache_exists(config: Dict[str, Any]) -> bool:
    """
    Check if processed data exists for this configuration.

    Args:
        config: Preprocessing configuration

    Returns:
        True if cache exists, False otherwise
    """
    train_cache = get_cache_path(config, 'train')
    test_cache = get_cache_path(config, 'test')

    return train_cache.exists() and test_cache.exists()


def load_from_cache(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data from cache.

    Args:
        config: Preprocessing configuration

    Returns:
        Tuple of (train_processed, test_processed)
    """
    train_cache = get_cache_path(config, 'train')
    test_cache = get_cache_path(config, 'test')

    print(f"📦 Loading preprocessed data from cache...")
    print(f"   Config version: {config.get('version', 'unknown')}")

    train_df = pd.read_csv(train_cache, parse_dates=['Date'])
    test_df = pd.read_csv(test_cache, parse_dates=['Date'])

    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")

    return train_df, test_df


def save_to_cache(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any]
) -> None:
    """
    Save preprocessed data to cache.

    Args:
        train_df: Preprocessed training data
        test_df: Preprocessed test data
        config: Preprocessing configuration
    """
    train_cache = get_cache_path(config, 'train')
    test_cache = get_cache_path(config, 'test')

    print(f"💾 Saving preprocessed data to cache...")
    print(f"   Config version: {config.get('version', 'unknown')}")

    # Save dataframes
    train_df.to_csv(train_cache, index=False)
    test_df.to_csv(test_cache, index=False)

    # Save config for reproducibility
    save_config_with_data(config, train_cache.parent)

    print(f"   ✅ Cache saved to: {train_cache.parent}")


def load_and_preprocess_data(
    config_name: str = 'baseline',
    use_cached: bool = True,
    force_preprocess: bool = False,
    data_dir: str = 'rossman_forecasting/data/raw',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess Rossmann data with caching support.

    Args:
        config_name: Name of preprocessing config
        use_cached: Use cached data if available
        force_preprocess: Force reprocessing even if cache exists
        data_dir: Directory with raw data
        verbose: Print progress messages

    Returns:
        Tuple of (train_processed, test_processed)
    """
    # Load config
    config = load_preprocessing_config(config_name)

    # Check cache
    cache_enabled = config.get('cache', {}).get('enabled', True)
    cache_exists = check_cache_exists(config)

    if cache_enabled and use_cached and not force_preprocess and cache_exists:
        return load_from_cache(config)

    # Load raw data
    train_raw, test_raw, store_df = load_raw_data(data_dir)

    # Apply preprocessing
    from .rossmann_features import apply_rossmann_preprocessing

    train_processed = apply_rossmann_preprocessing(
        train_raw, store_df, config, verbose=verbose
    )

    test_processed = apply_rossmann_preprocessing(
        test_raw, store_df, config, verbose=verbose
    )

    # Save to cache
    if cache_enabled:
        save_to_cache(train_processed, test_processed, config)

    return train_processed, test_processed


def list_available_configs() -> None:
    """
    List all available preprocessing configurations.
    """
    config_dir = Path('rossman_forecasting/configs/preprocessing')

    if not config_dir.exists():
        print("No configs found")
        return

    configs = sorted(config_dir.glob('*.yaml'))

    print("\n📋 Available Preprocessing Configs:")
    print("-" * 60)

    for config_file in configs:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        name = config_file.stem
        version = config.get('version', 'unknown')
        desc = config.get('description', 'No description')

        print(f"  {name:<20} (v{version})")
        print(f"    {desc}")

    print("-" * 60)


if __name__ == '__main__':
    # Test the data loader
    list_available_configs()
