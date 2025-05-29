"""Base agent implementation for AirAgent."""

from typing import Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.branch import RunnableBranch
from operator import itemgetter

from src.utils.nlp_utils import (
    extract_city,
    extract_date,
    generate_clarification_question,
)
from src.tools.listing_tools import return_listings_wrapper


class AirAgent:
    """Main agent class for AirAgent."""

    def __init__(self):
        """Initialize the AirAgent with LCEL chain."""
        self._build_chain()

    def _build_chain(self):
        """Build the LCEL chain for the agent."""
        # Convert functions to LCEL runnables
        city_extractor_runnable = RunnablePassthrough() | extract_city
        date_extractor_runnable = RunnablePassthrough() | extract_date
        clarification_runnable = RunnablePassthrough() | generate_clarification_question
        search_runnable = RunnablePassthrough() | return_listings_wrapper

        # Step 1: Extract City and Date in parallel and keep the original input
        extraction_step = RunnableParallel(
            {
                "city": RunnablePassthrough() | city_extractor_runnable,
                "date": RunnablePassthrough() | date_extractor_runnable,
                "original_query": RunnablePassthrough(),
            }
        )

        # Step 2: Check for missing info and prepare input for next step
        check_and_prepare_step = (
            RunnablePassthrough() | self._check_and_prepare_for_next_step
        )

        # Define a simple default runnable
        default_response_runnable = RunnablePassthrough() | (
            lambda x: "Sorry, I couldn't process your request with the provided information."
        )

        # Step 3: Define the branches including a default
        branch = RunnableBranch(
            (
                lambda x: x["action"] == "ask_clarification",
                itemgetter("input_for_tool") | clarification_runnable,
            ),
            (
                lambda x: x["action"] == "search",
                itemgetter("input_for_tool") | search_runnable,
            ),
            default_response_runnable,
        )

        # Combine the steps into a single LCEL chain
        self.chain = extraction_step | check_and_prepare_step | branch

    def _check_and_prepare_for_next_step(
        self, extraction_output: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Check for missing information and prepare input for next step.

        Args:
            extraction_output (Dict[str, str]): Output from extraction step

        Returns:
            Dict[str, Any]: Action and input for next step
        """
        city = extraction_output["city"]
        date = extraction_output["date"]
        original_query = extraction_output["original_query"]

        missing_parts = []

        # Check if city extraction failed or returned empty list
        if (
            not city
            or city == "No city found"
            or (isinstance(city, list) and len(city) == 0)
        ):
            missing_parts.append("city")
        if date == "No date found":
            missing_parts.append("date")

        if missing_parts:
            needed_info_str = " and ".join(missing_parts)
            return {"action": "ask_clarification", "input_for_tool": needed_info_str}
        else:
            return {
                "action": "search",
                "input_for_tool": {
                    "city": city,
                    "date": date,
                    "original_query": original_query,
                },
            }

    def invoke(self, query: str) -> str:
        """
        Process a user query and return a response.

        Args:
            query (str): User query

        Returns:
            str: Agent response
        """
        try:
            return self.chain.invoke(query)
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def get_chain(self):
        """Get the underlying LCEL chain for visualization or debugging."""
        return self.chain
