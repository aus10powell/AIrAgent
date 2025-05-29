"""Data loading utilities for AirAgent."""

import pandas as pd
from typing import Optional


def load_from_parquet(file_path: str) -> pd.DataFrame:
    """
    Load data from a parquet file.

    Args:
        file_path (str): Path to the parquet file

    Returns:
        pd.DataFrame: DataFrame containing the data
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()


def load_listings(city: str) -> pd.DataFrame:
    """
    Load listings data for a specific city.

    Args:
        city (str): City name

    Returns:
        pd.DataFrame: Listings data
    """
    from src.config import get_listings_path

    return load_from_parquet(get_listings_path(city))


def load_reviews(city: str) -> pd.DataFrame:
    """
    Load reviews data for a specific city.

    Args:
        city (str): City name

    Returns:
        pd.DataFrame: Reviews data
    """
    from src.config import get_reviews_path

    return load_from_parquet(get_reviews_path(city))
