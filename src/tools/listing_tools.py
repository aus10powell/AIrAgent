"""Listing search tools for AirAgent."""

from typing import Dict, Any, List
from langchain_core.tools import Tool
from src.utils.data_loader import load_listings
from src.config import SUPPORTED_CITIES


def return_listings(city: str) -> str:
    """
    Return top Airbnb listings for a specific city.

    Args:
        city (str): City name

    Returns:
        str: String representation of top listings
    """
    print(f"Returning listings for city: {city}")

    # Handle case where city is a list
    if isinstance(city, list):
        if not city:
            return "No city provided."
        city = city[0]

    city = city.lower().replace(" ", "_")

    if city not in SUPPORTED_CITIES:
        return f"No listings found for the city: {city}. Supported cities: {', '.join(SUPPORTED_CITIES)}"

    try:
        df_listings = load_listings(city)
        if df_listings.empty:
            return f"No listings data available for {city}"
        return df_listings.head(10).to_string()
    except Exception as e:
        return f"Error retrieving listings for {city}: {str(e)}"


def return_listings_wrapper(input_dict: Dict[str, Any]) -> str:
    """
    Wrapper function for return_listings to handle dictionary input.

    Args:
        input_dict (Dict[str, Any]): Dictionary containing city information

    Returns:
        str: Listings result
    """
    city = input_dict.get("city", "")
    return return_listings(city)


# Create the LangChain Tool
ListingSearchTool = Tool(
    name="search_listings",
    func=return_listings_wrapper,
    description="Search for Airbnb listings in a specific city. Input should be a dictionary with 'city' key.",
)
