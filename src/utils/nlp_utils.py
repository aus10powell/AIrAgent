"""NLP utilities for entity extraction."""

import spacy
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from src.config import OLLAMA_MODEL, DEFAULT_TEMPERATURE

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize LLM for city normalization
llm = Ollama(model=OLLAMA_MODEL, temperature=DEFAULT_TEMPERATURE)


def extract_city(text: str) -> List[str]:
    """
    Extracts city names from a user query and normalizes them using LLM.

    Args:
        text (str): Input text to extract cities from

    Returns:
        List[str]: List of normalized city names
    """
    doc = nlp(text)
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    print(f"Cities: {cities}")

    # Define the prompt template
    city_prompt_template = PromptTemplate.from_template(
        "Given the location '{raw_location}', return ONLY the name of the city it refers to. "
        "Respond with just the city name—no explanation, punctuation, or extra text."
    )

    # Create the LLM chain
    city_chain = city_prompt_template | llm

    if not cities:
        return []

    correct_cities = []
    for city in cities:
        result = city_chain.invoke({"raw_location": city})
        correct_cities.append(result.strip().lower().replace(" ", "_"))

    print(f"Correct cities: {correct_cities}")
    return correct_cities


def extract_date(text: str) -> str:
    """
    Extracts date information from a user query.

    Args:
        text (str): Input text to extract dates from

    Returns:
        str: Extracted date information or "No date found"
    """
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    print(f"Dates: {ents}")
    if ents:
        return str(ents)
    return "No date found"


def generate_clarification_question(missing_info_string: str) -> str:
    """
    Generates a clarifying question based on missing information.

    Args:
        missing_info_string (str): String describing what information is missing

    Returns:
        str: Clarification question
    """
    missing_info_string_lower = missing_info_string.lower()
    questions = []
    if "city" in missing_info_string_lower and "date" in missing_info_string_lower:
        questions.append("Which city and dates are you looking for?")
    elif "city" in missing_info_string_lower:
        questions.append("Which city are you looking for?")
    elif "date" in missing_info_string_lower:
        questions.append("What dates are you interested in?")
    else:
        questions.append("Could you please provide more details?")
    return " ".join(questions)
