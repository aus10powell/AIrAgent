"""Review analysis tools for AirAgent."""

from typing import List
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from src.config import OLLAMA_MODEL, SUMMARIZATION_TEMPERATURE

# Initialize LLM for review summarization
llm = Ollama(model=OLLAMA_MODEL, temperature=SUMMARIZATION_TEMPERATURE)


def summarize_reviews_tool_func(reviews: List[str]) -> str:
    """
    Summarizes a list of Airbnb reviews, identifying pros, cons,
    and determining which is more prominent in 3 sentences.

    Args:
        reviews (List[str]): List of review strings

    Returns:
        str: Summary of reviews
    """
    if not reviews:
        return "No reviews provided to summarize."

    # Combine reviews into a single string for the LLM
    reviews_text = "## Reviews:\n" + "\n---\n".join(reviews)

    # Define the prompt for the LLM
    summary_prompt_template = """
    Analyze the following Airbnb reviews provided below.

    Identify the main positive themes and negative themes mentioned across these reviews.

    Provide a summary that meets these requirements:
    1. It must be exactly 3 sentences long.
    2. The first sentence should summarize the main positive aspects mentioned by guests.
    3. The second sentence should summarize the main negative aspects mentioned by guests.
    4. The third sentence must state clearly whether the reviews were overall more positive or more negative, or if they were balanced, based on the prominence of pros vs. cons observed.

    Reviews:
    ---
    {reviews_content}
    ---

    Summary:
    """
    prompt = PromptTemplate.from_template(summary_prompt_template)

    # Create a chain to send the formatted reviews to the LLM
    summary_chain = prompt | llm

    # Invoke the chain to get the summary
    summary = summary_chain.invoke({"reviews_content": reviews_text})

    return summary.strip()


# Create the LangChain Tool
ReviewSummarizerTool = Tool(
    name="summarize_airbnb_reviews",
    func=summarize_reviews_tool_func,
    description="Summarizes a list of Airbnb reviews, highlighting pros, cons, and stating which is more prominent. Input should be a list of review strings.",
)
