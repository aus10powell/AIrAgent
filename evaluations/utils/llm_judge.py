"""LLM-as-Judge evaluation for RAG responses."""

import json
import re
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from src.config import OLLAMA_MODEL, DEFAULT_TEMPERATURE


class LLMJudge:
    """Evaluate RAG responses using LLM-as-judge approach."""

    def __init__(self, temperature: float = 0.1):
        """
        Initialize the LLM judge.

        Args:
            temperature (float): LLM temperature for consistent scoring
        """
        self.llm = Ollama(model=OLLAMA_MODEL, temperature=temperature)
        self.judge_prompt_template = self._create_judge_prompt()

    def _create_judge_prompt(self) -> PromptTemplate:
        """Create the structured prompt template for LLM judge."""
        template = """You are a wise and insightful expert in English and an expert in Airbnb listing information. Your task is to evaluate the quality of an answer to a user's question about an Airbnb listing.

Question: {query}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Please score the answer on these 4 metrics using a scale from 0.0 to 1.0:

Relevance: How relevant the answer is to the prompt
Completeness: How well the answer provides everything the user would benefit from knowing about the prompt
Intent: How well the answer satisfies the intent of the prompt  
Correctness: Based on your knowledge, how correct and factual the answer is

Important Instructions:
- Give scores as decimals between 0.0 and 1.0
- Be objective and consistent in your scoring
- Consider the context of Airbnb listings when evaluating
- If the actual answer contains errors or is completely irrelevant, score accordingly

Respond ONLY in this exact JSON format with no additional text:
{{
  "relevance": 0.0,
  "completeness": 0.0,
  "intent": 0.0,
  "correctness": 0.0
}}"""

        return PromptTemplate.from_template(template)

    def evaluate_response(
        self, query: str, expected_answer: str, actual_answer: str, max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG response using LLM judge.

        Args:
            query (str): The original query
            expected_answer (str): Expected/ground truth answer
            actual_answer (str): RAG-generated answer
            max_retries (int): Maximum number of retry attempts

        Returns:
            Dict[str, Any]: Judge scores and metadata
        """
        # Format the prompt
        formatted_prompt = self.judge_prompt_template.format(
            query=query,
            expected_answer=expected_answer or "Not specified",
            actual_answer=actual_answer,
        )

        # Try to get a valid response with retries
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Get LLM response
                response = self.llm.generate([formatted_prompt])

                if hasattr(response, "generations") and response.generations:
                    if response.generations[0] and hasattr(
                        response.generations[0][0], "text"
                    ):
                        raw_response = response.generations[0][0].text.strip()
                    else:
                        raise ValueError("Could not extract text from LLM response")
                else:
                    raise ValueError("No generations in LLM response")

                # Parse the JSON response
                scores = self._parse_judge_response(raw_response)

                if scores:
                    # Calculate average score
                    avg_score = sum(scores.values()) / len(scores)

                    return {
                        "llm_judge_relevance": scores["relevance"],
                        "llm_judge_completeness": scores["completeness"],
                        "llm_judge_intent": scores["intent"],
                        "llm_judge_correctness": scores["correctness"],
                        "llm_judge_average": avg_score,
                        "llm_judge_raw_response": raw_response,
                        "llm_judge_error": None,
                        "llm_judge_attempts": attempt + 1,
                    }

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    continue  # Try again

        # If all attempts failed, return error state
        return {
            "llm_judge_relevance": None,
            "llm_judge_completeness": None,
            "llm_judge_intent": None,
            "llm_judge_correctness": None,
            "llm_judge_average": None,
            "llm_judge_raw_response": None,
            "llm_judge_error": last_error,
            "llm_judge_attempts": max_retries + 1,
        }

    def _parse_judge_response(self, response: str) -> Optional[Dict[str, float]]:
        """
        Parse the LLM judge response to extract scores.

        Args:
            response (str): Raw LLM response

        Returns:
            Optional[Dict[str, float]]: Parsed scores or None if parsing failed
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{[^{}]*\}", response)
            if json_match:
                json_str = json_match.group()
                scores = json.loads(json_str)

                # Validate the expected keys exist
                required_keys = ["relevance", "completeness", "intent", "correctness"]
                if all(key in scores for key in required_keys):
                    # Validate scores are numbers between 0 and 1
                    validated_scores = {}
                    for key in required_keys:
                        score = float(scores[key])
                        if 0.0 <= score <= 1.0:
                            validated_scores[key] = score
                        else:
                            # Clamp to valid range
                            validated_scores[key] = max(0.0, min(1.0, score))

                    return validated_scores

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Try alternative parsing strategies
            return self._fallback_parse(response)

        return None

    def _fallback_parse(self, response: str) -> Optional[Dict[str, float]]:
        """
        Fallback parsing for when JSON parsing fails.

        Args:
            response (str): Raw LLM response

        Returns:
            Optional[Dict[str, float]]: Parsed scores or None
        """
        try:
            # Try to extract numbers from the response
            scores = {}
            metrics = ["relevance", "completeness", "intent", "correctness"]

            for metric in metrics:
                # Look for patterns like "relevance: 0.8" or "relevance": 0.8
                pattern = rf'{metric}["\']?\s*:?\s*([0-9]*\.?[0-9]+)'
                match = re.search(pattern, response.lower())
                if match:
                    score = float(match.group(1))
                    scores[metric] = max(0.0, min(1.0, score))  # Clamp to 0-1

            # Only return if we found all 4 metrics
            if len(scores) == 4:
                return scores

        except (ValueError, AttributeError):
            pass

        return None

    def evaluate_batch(
        self, evaluations: list, use_judge: bool = True, progress_callback=None
    ) -> list:
        """
        Evaluate a batch of responses with optional progress tracking.

        Args:
            evaluations (list): List of evaluation dictionaries
            use_judge (bool): Whether to use LLM judge
            progress_callback: Function to call with progress updates

        Returns:
            list: Updated evaluations with judge scores
        """
        if not use_judge:
            return evaluations

        total = len(evaluations)
        for i, eval_data in enumerate(evaluations):
            if progress_callback:
                progress_callback(i + 1, total)

            # Skip if this evaluation already has an error
            if eval_data.get("error"):
                continue

            judge_scores = self.evaluate_response(
                query=eval_data.get("query", ""),
                expected_answer=eval_data.get("expected_answer", ""),
                actual_answer=eval_data.get("actual_answer", ""),
            )

            # Add judge scores to the evaluation
            eval_data.update(judge_scores)

        return evaluations
