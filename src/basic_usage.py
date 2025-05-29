"""Basic usage example for AirAgent."""

import random
from src.agents import AirAgent
from src.tools import ReviewSummarizerTool, RAGTool
from src.utils.data_loader import load_listings


def main():
    """Demonstrate basic AirAgent usage."""

    # Initialize the agent
    agent = AirAgent()

    print("=== AirAgent Basic Usage Example ===\n")

    # Example 1: Query with city and date
    print("1. Query with city:")
    query1 = "I'm looking for a place to stay in Seattle next weekend."
    response1 = agent.invoke(query1)
    print(f"Query: {query1}")
    print(f"Response: {response1}\n")

    # Example 2: Query missing city
    print("2. Query missing city:")
    query2 = "I need accommodation for next month."
    response2 = agent.invoke(query2)
    print(f"Query: {query2}")
    print(f"Response: {response2}\n")

    # Example 3: Using review summarizer tool directly
    print("3. Review summarizer tool using made up reviews:")
    sample_reviews = [
        "Great location, very clean, but the internet was slow.",
        "Host was amazing, made me feel at home. The room was small though.",
        "Cool place, unique neighborhood, but noisy at night.",
        "Loved the art and community! Bed was super comfortable.",
        "Conveniently located. Shower was a bit tricky to figure out.",
    ]

    summary = ReviewSummarizerTool.func(sample_reviews)
    print(f"Reviews: {sample_reviews}")
    print(f"Summary: {summary}\n")

    # Example 4: Using RAG tool directly
    print("4. RAG tool with sample content:")
    sample_listing_content = """
    This cozy apartment is located in downtown Seattle with stunning views of the Space Needle. 
    The apartment features a queen-sized bed that is extremely comfortable with premium linens.
    There is high-speed WiFi available throughout the apartment.
    The kitchen is fully equipped with modern appliances including a dishwasher and coffee maker.
    """

    rag_input = {"query": "What size is the bed?", "corpus": sample_listing_content}

    rag_response = RAGTool.func(rag_input)
    print(f"Query: {rag_input['query']}")
    print(f"Response: {rag_response}\n")

    # Example 5: Load real Seattle listing and query about kitchen
    print("5. Real Seattle listing kitchen analysis:")
    try:
        # Load Seattle listings
        seattle_listings = load_listings("seattle")

        if not seattle_listings.empty:
            # Get a random listing that has description data
            listings_with_desc = seattle_listings.dropna(subset=["description"])

            if not listings_with_desc.empty:
                random_listing = listings_with_desc.sample(n=1).iloc[0]

                # Extract relevant information for RAG
                listing_info = f"""
                Listing Name: {random_listing.get('name', 'N/A')}
                Description: {random_listing.get('description', 'N/A')}
                Amenities: {random_listing.get('amenities', 'N/A')}
                Property Type: {random_listing.get('property_type', 'N/A')}
                Room Type: {random_listing.get('room_type', 'N/A')}
                """

                # Query about the kitchen
                kitchen_query = {
                    "query": "What kind of kitchen does this listing have? Describe the kitchen amenities and features.",
                    "corpus": listing_info,
                }

                kitchen_response = RAGTool.func(kitchen_query)

                print(f"Listing: {random_listing.get('name', 'Unknown')}")
                print(
                    f"Location: {random_listing.get('neighbourhood_cleansed', 'Unknown')}"
                )
                print(f"Price: ${random_listing.get('price', 'N/A')}")
                print(f"Query: {kitchen_query['query']}")
                print(f"Kitchen Analysis: {kitchen_response}")
            else:
                print("No listings with descriptions found in the dataset.")
        else:
            print("No Seattle listings data available.")

    except Exception as e:
        print(f"Error loading Seattle listings: {e}")
        print(
            "Make sure the Seattle listings data is available and the path in config.py is correct."
        )


if __name__ == "__main__":
    main()
