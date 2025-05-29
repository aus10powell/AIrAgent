"""Basic usage example for AirAgent."""

from src.agents import AirAgent
from src.tools import ReviewSummarizerTool, RAGTool


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
    print("4. RAG tool:")
    sample_listing_content = """
    This cozy apartment is located in downtown Seattle with stunning views of the Space Needle. 
    The apartment features a queen-sized bed that is extremely comfortable with premium linens.
    There is high-speed WiFi available throughout the apartment.
    The kitchen is fully equipped with modern appliances including a dishwasher and coffee maker.
    """

    rag_input = {"query": "What size is the bed?", "corpus": sample_listing_content}

    rag_response = RAGTool.func(rag_input)
    print(f"Query: {rag_input['query']}")
    print(f"Response: {rag_response}")


if __name__ == "__main__":
    main()
