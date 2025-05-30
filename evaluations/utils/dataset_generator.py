"""Dataset generator for RAG evaluation."""

import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from src.utils.data_loader import load_listings
from src.config import SUPPORTED_CITIES


class DatasetGenerator:
    """Generate evaluation datasets for RAG testing."""

    def __init__(self, output_dir: str = "evaluations/datasets/rag_evaluation"):
        """
        Initialize the dataset generator.

        Args:
            output_dir (str): Directory to save generated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_relevant_chunks(
        self, listing: pd.Series, chunk_size: int = 100
    ) -> List[str]:
        """
        Extract relevant text chunks from a listing for RAG evaluation.

        Args:
            listing (pd.Series): Listing data
            chunk_size (int): Target chunk size in words

        Returns:
            List[str]: List of text chunks
        """
        # Combine relevant listing information
        text_parts = []

        # Add listing name
        if pd.notna(listing.get("name")):
            text_parts.append(f"Listing Name: {listing['name']}")

        # Add description
        if pd.notna(listing.get("description")):
            text_parts.append(f"Description: {listing['description']}")

        # Add amenities
        if pd.notna(listing.get("amenities")):
            text_parts.append(f"Amenities: {listing['amenities']}")

        # Add property details
        property_details = []
        for field in [
            "property_type",
            "room_type",
            "bedrooms",
            "bathrooms",
            "accommodates",
        ]:
            if pd.notna(listing.get(field)):
                property_details.append(f"{field}: {listing[field]}")

        if property_details:
            text_parts.append("Property Details: " + ", ".join(property_details))

        # Add location info
        location_details = []
        for field in ["neighbourhood_cleansed", "neighbourhood_group_cleansed"]:
            if pd.notna(listing.get(field)):
                location_details.append(f"{field}: {listing[field]}")

        if location_details:
            text_parts.append("Location: " + ", ".join(location_details))

        # Combine all text
        full_text = " ".join(text_parts)

        # Simple chunking - split into roughly equal sized chunks
        words = full_text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += 1

            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        # Add remaining words as final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_test_queries(self, listing: pd.Series) -> List[Dict[str, Any]]:
        """
        Generate test queries for a listing.

        Args:
            listing (pd.Series): Listing data

        Returns:
            List[Dict[str, Any]]: List of query dictionaries
        """
        queries = []

        # Query templates with expected answer patterns
        query_templates = [
            {
                "query": "What type of property is this?",
                "expected_field": "property_type",
                "category": "property_details",
            },
            {
                "query": "How many bedrooms does this place have?",
                "expected_field": "bedrooms",
                "category": "property_details",
            },
            {
                "query": "What amenities are available?",
                "expected_field": "amenities",
                "category": "amenities",
            },
            {
                "query": "What neighborhood is this located in?",
                "expected_field": "neighbourhood_cleansed",
                "category": "location",
            },
            {
                "query": "How many people can this accommodate?",
                "expected_field": "accommodates",
                "category": "property_details",
            },
            {
                "query": "What is the room type?",
                "expected_field": "room_type",
                "category": "property_details",
            },
        ]

        for template in query_templates:
            field = template["expected_field"]
            if pd.notna(listing.get(field)):
                queries.append(
                    {
                        "query": template["query"],
                        "expected_answer": str(listing[field]),
                        "category": template["category"],
                        "listing_id": listing.get("id", "unknown"),
                    }
                )

        return queries

    def create_rag_evaluation_dataset(
        self, city: str, num_listings: int = 100, chunk_size: int = 100, seed: int = 42
    ) -> Tuple[str, str]:
        """
        Create a RAG evaluation dataset with chunks and queries.

        Args:
            city (str): City to generate dataset for
            num_listings (int): Number of listings to include
            chunk_size (int): Target chunk size in words
            seed (int): Random seed for reproducibility

        Returns:
            Tuple[str, str]: Paths to chunks file and queries file
        """
        random.seed(seed)

        # Load listings data
        listings_df = load_listings(city)

        if listings_df.empty:
            raise ValueError(f"No listings data found for city: {city}")

        # Filter listings with sufficient data
        required_fields = ["name", "description"]
        valid_listings = listings_df.dropna(subset=required_fields)

        if len(valid_listings) < num_listings:
            print(
                f"Warning: Only {len(valid_listings)} valid listings found, using all available"
            )
            num_listings = len(valid_listings)

        # Sample listings
        sampled_listings = valid_listings.sample(n=num_listings, random_state=seed)

        # Generate chunks and queries
        all_chunks = []
        all_queries = []

        for idx, (_, listing) in enumerate(sampled_listings.iterrows()):
            # Extract chunks
            chunks = self.extract_relevant_chunks(listing, chunk_size)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_data = {
                    "chunk_id": f"{city}_{listing.get('id', idx)}_{chunk_idx}",
                    "listing_id": listing.get("id", f"listing_{idx}"),
                    "city": city,
                    "chunk_text": chunk,
                    "chunk_index": chunk_idx,
                    "listing_name": listing.get("name", "Unknown"),
                }
                all_chunks.append(chunk_data)

            # Generate queries for this listing
            queries = self.generate_test_queries(listing)
            for query in queries:
                query["chunks"] = [
                    chunk["chunk_id"]
                    for chunk in all_chunks
                    if chunk["listing_id"] == listing.get("id", f"listing_{idx}")
                ]

            all_queries.extend(queries)

        # Save datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks_filename = f"chunks_{city}_{num_listings}listings_{timestamp}.json"
        queries_filename = f"queries_{city}_{num_listings}listings_{timestamp}.json"

        chunks_path = self.output_dir / chunks_filename
        queries_path = self.output_dir / queries_filename

        # Save chunks
        with open(chunks_path, "w") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        # Save queries
        with open(queries_path, "w") as f:
            json.dump(all_queries, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(all_chunks)} chunks and {len(all_queries)} queries")
        print(f"Chunks saved to: {chunks_path}")
        print(f"Queries saved to: {queries_path}")

        return str(chunks_path), str(queries_path)

    def load_evaluation_dataset(
        self, chunks_file: str, queries_file: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load an existing evaluation dataset.

        Args:
            chunks_file (str): Path to chunks file
            queries_file (str): Path to queries file

        Returns:
            Tuple[List[Dict], List[Dict]]: Chunks and queries data
        """
        with open(chunks_file, "r") as f:
            chunks = json.load(f)

        with open(queries_file, "r") as f:
            queries = json.load(f)

        return chunks, queries
