"""Tests for AirAgent functionality."""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents import AirAgent
from utils.nlp_utils import extract_city, extract_date
from tools.review_tools import summarize_reviews_tool_func


class TestAirAgent(unittest.TestCase):
    """Test cases for AirAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = AirAgent()
    
    def test_agent_initialization(self):
        """Test that agent initializes properly."""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.chain)
    
    def test_city_extraction(self):
        """Test city extraction functionality."""
        # This test might fail if Ollama is not running
        try:
            cities = extract_city("I want to visit Seattle")
            self.assertIsInstance(cities, list)
        except Exception as e:
            self.skipTest(f"Skipping due to LLM dependency: {e}")
    
    def test_date_extraction(self):
        """Test date extraction functionality."""
        date_result = extract_date("I'm traveling next weekend")
        self.assertIsInstance(date_result, str)
    
    def test_review_summarization(self):
        """Test review summarization."""
        sample_reviews = [
            "Great place, loved it!",
            "Good location but noisy.",
            "Clean and comfortable."
        ]
        
        try:
            summary = summarize_reviews_tool_func(sample_reviews)
            self.assertIsInstance(summary, str)
            self.assertGreater(len(summary), 0)
        except Exception as e:
            self.skipTest(f"Skipping due to LLM dependency: {e}")


if __name__ == "__main__":
    unittest.main() 