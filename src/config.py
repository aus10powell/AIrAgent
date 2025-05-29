"""Configuration settings for AirAgent."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path("/Users/aus10powell/Documents/Projects/AirRanker/data")

# Model settings
OLLAMA_MODEL = "llama3.2"
DEFAULT_TEMPERATURE = 0.5
SUMMARIZATION_TEMPERATURE = 0.2

# Supported cities
SUPPORTED_CITIES = ["seattle", "san_francisco"]


# File paths
def get_listings_path(city: str) -> str:
    """Get the path to listings parquet file for a city."""
    return str(DATA_ROOT / city / "listings.parquet")


def get_reviews_path(city: str) -> str:
    """Get the path to reviews parquet file for a city."""
    return str(DATA_ROOT / city / "reviews.parquet")


# RAG settings
DEFAULT_CHUNK_SIZE = 100
