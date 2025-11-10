"""
Rossmann-specific feature engineering.

IMPORTANT: This module only adds Rossmann-specific features.
Time-series features (date encoding, lags, rolling stats) are handled by tf_predictor.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


def merge_store_data(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge store metadata with sales data.

    Args:
        df: Sales data (train or test)
        store_df: Store metadata

    Returns:
        Merged DataFrame
    """
    return df.merge(store_df, on='Store', how='left')


def calculate_months_since(df: pd.DataFrame, year_col: str, month_col: str, ref_date_col: str = 'Date') -> pd.Series:
    """
    Calculate months since a given year/month.

    Args:
        df: DataFrame
        year_col: Column name for year
        month_col: Column name for month
        ref_date_col: Reference date column

    Returns:
        Series with months since
    """
    # Convert reference date to datetime if string
    if df[ref_date_col].dtype == 'object':
        ref_dates = pd.to_datetime(df[ref_date_col])
    else:
        ref_dates = df[ref_date_col]

    # Handle missing year/month (fill with large value = no competition/promo yet)
    years = df[year_col].fillna(3000).astype(int)
    months = df[month_col].fillna(1).astype(int)

    # Create date from year/month
    event_dates = pd.to_datetime(years.astype(str) + '-' + months.astype(str) + '-01', errors='coerce')

    # Calculate months difference
    months_since = ((ref_dates.dt.year - event_dates.dt.year) * 12 +
                    (ref_dates.dt.month - event_dates.dt.month))

    # Set to 0 if event hasn't happened yet (future date)
    months_since = months_since.clip(lower=0)

    # Set to 999 if date is invalid (missing year/month)
    months_since = months_since.fillna(999)

    return months_since


def create_competition_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create competition-related features.

    Args:
        df: DataFrame with store data merged
        config: Configuration dictionary

    Returns:
        DataFrame with competition features
    """
    if not config.get('enabled', True):
        return df

    df = df.copy()

    # Fill missing competition distance
    if 'CompetitionDistance' in df.columns:
        fill_method = config.get('fill_distance_missing', 'median')
        if fill_method == 'median':
            df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
        elif fill_method == 'max':
            df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].max())
        else:
            df['CompetitionDistance'] = df['CompetitionDistance'].fillna(0)

    # Calculate months since competition opened
    if config.get('calculate_months_since', True):
        df['MonthsSinceCompetition'] = calculate_months_since(
            df, 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'
        )

    return df


def create_promotion_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create promotion-related features.

    Args:
        df: DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with promotion features
    """
    if not config.get('enabled', True):
        return df

    df = df.copy()

    # Promo2 features
    if config.get('use_promo2', True) and 'Promo2' in df.columns:
        # Months since Promo2 started
        if config.get('calculate_months_since_promo2', True):
            df['MonthsSincePromo2'] = calculate_months_since(
                df, 'Promo2SinceYear', 'Promo2SinceWeek'
            )

        # Is current month in PromoInterval
        if config.get('create_promo_month_indicator', True) and 'PromoInterval' in df.columns:
            df['IsPromoMonth'] = 0

            # Convert date to month name
            df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%b')

            # Check if month is in PromoInterval
            for idx, row in df.iterrows():
                if pd.notna(row['PromoInterval']) and pd.notna(row['Month']):
                    promo_months = row['PromoInterval'].split(',')
                    df.loc[idx, 'IsPromoMonth'] = int(row['Month'] in promo_months)

            df = df.drop('Month', axis=1)

    return df


def encode_holidays(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Encode holiday features.

    Args:
        df: DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with encoded holidays
    """
    if not config.get('enabled', True):
        return df

    df = df.copy()

    # State Holiday: a/b/c/0
    if config.get('encode_state_holiday', True) and 'StateHoliday' in df.columns:
        # Convert to string and one-hot encode
        df['StateHoliday'] = df['StateHoliday'].astype(str)
        state_holiday_dummies = pd.get_dummies(df['StateHoliday'], prefix='StateHoliday')
        df = pd.concat([df, state_holiday_dummies], axis=1)
        df = df.drop('StateHoliday', axis=1)

    # School Holiday is already binary (0/1), keep as is
    # Will be handled by tf_predictor's automatic feature detection

    return df


def encode_store_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Encode store categorical features.

    Args:
        df: DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with encoded store features
    """
    if not config.get('enabled', True):
        return df

    df = df.copy()

    # Store Type: a/b/c/d
    if config.get('encode_store_type', True) and 'StoreType' in df.columns:
        store_type_dummies = pd.get_dummies(df['StoreType'], prefix='StoreType')
        df = pd.concat([df, store_type_dummies], axis=1)
        df = df.drop('StoreType', axis=1)

    # Assortment: a/b/c
    if config.get('encode_assortment', True) and 'Assortment' in df.columns:
        assortment_dummies = pd.get_dummies(df['Assortment'], prefix='Assortment')
        df = pd.concat([df, assortment_dummies], axis=1)
        df = df.drop('Assortment', axis=1)

    return df


def create_domain_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create domain-specific engineered features.

    Args:
        df: DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with domain features
    """
    df = df.copy()

    # Sales per customer (only for train data with Customers column)
    if config.get('sales_per_customer', True) and 'Customers' in df.columns and 'Sales' in df.columns:
        df['SalesPerCustomer'] = df['Sales'] / df['Customers'].replace(0, 1)

    # Indicator for stores with customer data
    if config.get('customers_indicator', True) and 'Customers' in df.columns:
        df['HasCustomerData'] = (df['Customers'] > 0).astype(int)

    return df


def filter_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply data filtering based on configuration.

    Args:
        df: DataFrame
        config: Configuration dictionary

    Returns:
        Filtered DataFrame
    """
    df = df.copy()

    # Remove closed stores
    if config.get('remove_closed_stores', True) and 'Open' in df.columns:
        initial_len = len(df)
        df = df[df['Open'] == 1]
        if initial_len > len(df):
            print(f"   Filtered out {initial_len - len(df)} closed store records")

    # Remove zero sales (optional, not recommended for validation)
    if config.get('remove_zero_sales', False) and 'Sales' in df.columns:
        initial_len = len(df)
        df = df[df['Sales'] > 0]
        if initial_len > len(df):
            print(f"   Filtered out {initial_len - len(df)} zero sales records")

    return df


def apply_rossmann_preprocessing(
    df: pd.DataFrame,
    store_df: pd.DataFrame,
    config: Dict[str, Any],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply all Rossmann-specific preprocessing steps.

    This only adds Rossmann domain features. Time-series features
    (date features, lags, rolling stats) are added by tf_predictor.

    Args:
        df: Sales data (train or test)
        store_df: Store metadata
        config: Preprocessing configuration
        verbose: Print progress messages

    Returns:
        Preprocessed DataFrame
    """
    if verbose:
        print(f"🔧 Applying Rossmann preprocessing: {config.get('version', 'unknown')}")
        print(f"   Initial shape: {df.shape}")

    # Step 1: Merge store data
    df = merge_store_data(df, store_df)
    if verbose:
        print(f"   After merge: {df.shape}")

    # Step 2: Create competition features
    df = create_competition_features(df, config.get('competition', {}))

    # Step 3: Create promotion features
    df = create_promotion_features(df, config.get('promotion', {}))

    # Step 4: Encode holidays
    df = encode_holidays(df, config.get('holidays', {}))

    # Step 5: Encode store features
    df = encode_store_features(df, config.get('store', {}))

    # Step 6: Create domain features
    df = create_domain_features(df, config.get('domain_features', {}))

    # Step 7: Apply filtering
    df = filter_data(df, config.get('filtering', {}))

    # Fill any remaining NaN values
    df = df.fillna(0)

    if verbose:
        print(f"   Final shape: {df.shape}")
        print(f"   Total features: {len(df.columns)}")

    return df
