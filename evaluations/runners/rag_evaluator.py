"""RAG evaluation runner for AirAgent."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from src.tools.rag_tools import simple_rag_with_ollama
from evaluations.utils.llm_judge import LLMJudge


class RAGEvaluator:
    """Evaluate RAG performance using generated datasets."""

    def __init__(self, results_dir: str = "evaluations/results/rag_evaluation"):
        """
        Initialize the RAG evaluator.

        Args:
            results_dir (str): Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.llm_judge = None  # Initialize lazily

    def _get_llm_judge(self) -> LLMJudge:
        """Get or create LLM judge instance (lazy initialization)."""
        if self.llm_judge is None:
            self.llm_judge = LLMJudge()
        return self.llm_judge

    def evaluate_rag_response(
        self, query: str, expected_answer: str, actual_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG response.

        Args:
            query (str): The input query
            expected_answer (str): Expected answer
            actual_answer (str): RAG-generated answer

        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Simple evaluation metrics
        metrics = {
            "query": query,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "response_length": len(actual_answer),
            "contains_expected": (
                expected_answer.lower() in actual_answer.lower()
                if expected_answer
                else False
            ),
            "is_error": actual_answer.startswith("[Error]") or "Error" in actual_answer,
            "is_empty": len(actual_answer.strip()) == 0,
        }

        # Calculate simple relevance score
        if expected_answer and not metrics["is_error"] and not metrics["is_empty"]:
            # Count word overlap
            expected_words = set(expected_answer.lower().split())
            actual_words = set(actual_answer.lower().split())

            if expected_words:
                word_overlap = len(expected_words.intersection(actual_words)) / len(
                    expected_words
                )
                metrics["word_overlap_score"] = word_overlap
            else:
                metrics["word_overlap_score"] = 0.0
        else:
            metrics["word_overlap_score"] = 0.0

        return metrics

    def run_evaluation(
        self,
        chunks_file: str,
        queries_file: str,
        chunk_size: int = 100,
        max_queries: int = None,
        use_llm_judge: bool = False,
    ) -> str:
        """
        Run evaluation on a dataset.

        Args:
            chunks_file (str): Path to chunks JSON file
            queries_file (str): Path to queries JSON file
            chunk_size (int): Chunk size for RAG
            max_queries (int): Maximum number of queries to evaluate (for testing)
            use_llm_judge (bool): Whether to include LLM judge evaluation

        Returns:
            str: Path to results file
        """
        print(f"Loading evaluation dataset...")

        # Load datasets
        with open(chunks_file, "r") as f:
            chunks_data = json.load(f)

        with open(queries_file, "r") as f:
            queries_data = json.load(f)

        # Create chunks lookup
        chunks_lookup = {chunk["chunk_id"]: chunk for chunk in chunks_data}

        # Limit queries for testing if specified
        if max_queries:
            queries_data = queries_data[:max_queries]

        print(f"Evaluating {len(queries_data)} queries...")
        if use_llm_judge:
            print("🧠 LLM Judge evaluation enabled (this will take longer)")

        results = []
        start_time = time.time()

        for i, query_data in enumerate(queries_data):
            print(
                f"Processing query {i+1}/{len(queries_data)}: {query_data['query'][:50]}..."
            )

            try:
                # Get relevant chunks for this query
                relevant_chunks = []
                for chunk_id in query_data.get("chunks", []):
                    if chunk_id in chunks_lookup:
                        relevant_chunks.append(chunks_lookup[chunk_id]["chunk_text"])

                # Combine chunks into corpus
                corpus = " ".join(relevant_chunks)

                if not corpus.strip():
                    print(f"Warning: No corpus found for query {i+1}")
                    actual_answer = "No content available to search through."
                else:
                    # Run RAG
                    actual_answer = simple_rag_with_ollama(
                        query=query_data["query"], corpus=corpus, chunk_size=chunk_size
                    )

                # Evaluate response with traditional metrics
                metrics = self.evaluate_rag_response(
                    query=query_data["query"],
                    expected_answer=query_data.get("expected_answer", ""),
                    actual_answer=actual_answer,
                )

                # Add metadata
                result = {
                    "query_id": i,
                    "listing_id": query_data.get("listing_id"),
                    "category": query_data.get("category"),
                    "num_chunks": len(relevant_chunks),
                    "corpus_length": len(corpus),
                    **metrics,
                }

                # Add LLM judge evaluation if requested
                if use_llm_judge and not result.get("is_error", False):
                    print(f"  🧠 Running LLM judge for query {i+1}...")
                    judge_scores = self._get_llm_judge().evaluate_response(
                        query=query_data["query"],
                        expected_answer=query_data.get("expected_answer", ""),
                        actual_answer=actual_answer,
                    )
                    result.update(judge_scores)

                results.append(result)

            except Exception as e:
                print(f"Error processing query {i+1}: {str(e)}")
                error_result = {
                    "query_id": i,
                    "query": query_data["query"],
                    "error": str(e),
                    "listing_id": query_data.get("listing_id"),
                    "category": query_data.get("category"),
                }
                results.append(error_result)

        # Calculate summary statistics
        successful_results = [r for r in results if "error" not in r]

        if successful_results:
            summary = {
                "total_queries": len(queries_data),
                "successful_evaluations": len(successful_results),
                "error_rate": (len(results) - len(successful_results)) / len(results),
                "avg_response_length": sum(
                    r.get("response_length", 0) for r in successful_results
                )
                / len(successful_results),
                "contains_expected_rate": sum(
                    r.get("contains_expected", False) for r in successful_results
                )
                / len(successful_results),
                "avg_word_overlap_score": sum(
                    r.get("word_overlap_score", 0) for r in successful_results
                )
                / len(successful_results),
                "error_response_rate": sum(
                    r.get("is_error", False) for r in successful_results
                )
                / len(successful_results),
                "empty_response_rate": sum(
                    r.get("is_empty", False) for r in successful_results
                )
                / len(successful_results),
            }

            # Add LLM judge summary statistics if available
            if use_llm_judge:
                llm_judge_results = [
                    r
                    for r in successful_results
                    if r.get("llm_judge_relevance") is not None
                ]

                if llm_judge_results:
                    summary.update(
                        {
                            "llm_judge_success_rate": len(llm_judge_results)
                            / len(successful_results),
                            "avg_llm_judge_relevance": sum(
                                r.get("llm_judge_relevance", 0)
                                for r in llm_judge_results
                            )
                            / len(llm_judge_results),
                            "avg_llm_judge_completeness": sum(
                                r.get("llm_judge_completeness", 0)
                                for r in llm_judge_results
                            )
                            / len(llm_judge_results),
                            "avg_llm_judge_intent": sum(
                                r.get("llm_judge_intent", 0) for r in llm_judge_results
                            )
                            / len(llm_judge_results),
                            "avg_llm_judge_correctness": sum(
                                r.get("llm_judge_correctness", 0)
                                for r in llm_judge_results
                            )
                            / len(llm_judge_results),
                            "avg_llm_judge_overall": sum(
                                r.get("llm_judge_average", 0) for r in llm_judge_results
                            )
                            / len(llm_judge_results),
                        }
                    )
        else:
            summary = {
                "total_queries": len(queries_data),
                "successful_evaluations": 0,
                "error_rate": 1.0,
            }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        judge_suffix = "_with_judge" if use_llm_judge else ""
        results_filename = f"rag_evaluation{judge_suffix}_{timestamp}.json"
        results_path = self.results_dir / results_filename

        evaluation_data = {
            "metadata": {
                "chunks_file": chunks_file,
                "queries_file": queries_file,
                "chunk_size": chunk_size,
                "max_queries": max_queries,
                "use_llm_judge": use_llm_judge,
                "evaluation_timestamp": timestamp,
                "total_runtime_seconds": time.time() - start_time,
            },
            "summary": summary,
            "results": results,
        }

        with open(results_path, "w") as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation completed!")
        print(f"Results saved to: {results_path}")
        print(f"Summary statistics:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        return str(results_path)

    def compare_evaluations(self, results_files: List[str]) -> str:
        """
        Compare multiple evaluation results.

        Args:
            results_files (List[str]): List of result file paths

        Returns:
            str: Path to comparison report
        """
        comparisons = []

        for file_path in results_files:
            with open(file_path, "r") as f:
                data = json.load(f)

            comparison_data = {
                "file": file_path,
                "timestamp": data["metadata"]["evaluation_timestamp"],
                "use_llm_judge": data["metadata"].get("use_llm_judge", False),
                "summary": data["summary"],
            }
            comparisons.append(comparison_data)

        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = f"evaluation_comparison_{timestamp}.json"
        comparison_path = self.results_dir / comparison_filename

        with open(comparison_path, "w") as f:
            json.dump(comparisons, f, indent=2, ensure_ascii=False)

        print(f"Comparison saved to: {comparison_path}")
        return str(comparison_path)
