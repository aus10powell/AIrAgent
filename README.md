# AirAgent

AI-powered Airbnb listing assistant that helps users find accommodations and analyze reviews using natural language processing and retrieval-augmented generation (RAG).

## Features

- **Natural Language Query Processing**: Extract cities and dates from user queries
- **Listing Search**: Find Airbnb listings in supported cities (Seattle, San Francisco)
- **Review Summarization**: Analyze and summarize Airbnb reviews with pros/cons
- **RAG-based Q&A**: Answer questions about specific listings using retrieval-augmented generation
- **Modular Architecture**: Clean, maintainable code structure with separate concerns

## Project Structure

```
AirAgent/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base_agent.py         # Main agent implementation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── listing_tools.py      # Listing search functionality
│   │   ├── review_tools.py       # Review summarization
│   │   └── rag_tools.py          # RAG-based Q&A
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py        # Data loading utilities
│       ├── nlp_utils.py          # NLP processing functions
│       └── date_utils.py         # Date handling utilities
├── examples/
│   └── basic_usage.py            # Usage examples
├── tests/
│   ├── __init__.py
│   └── test_agent.py             # Unit tests
├── base_agent.ipynb             # Original notebook (for reference)
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AirAgent.git
cd AirAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Prerequisites

- **Ollama**: Make sure Ollama is installed and running with the `llama3.2` model
- **Data**: Update the data paths in `src/config.py` to point to your Airbnb data files

## Quick Start

```python
from src.agents import AirAgent

# Initialize the agent
agent = AirAgent()

# Query for listings
response = agent.invoke("I'm looking for a place to stay in Seattle next weekend.")
print(response)
```

## Usage Examples

### Basic Agent Usage

```python
from src.agents import AirAgent

agent = AirAgent()

# Query with city
response = agent.invoke("Find me accommodation in Seattle for next month")

# Query missing information (will ask for clarification)
response = agent.invoke("I need a place to stay")
```

### Using Tools Directly

```python
from src.tools import ReviewSummarizerTool, RAGTool

# Summarize reviews
reviews = [
    "Great location, very clean, but the internet was slow.",
    "Host was amazing, made me feel at home. The room was small though.",
]
summary = ReviewSummarizerTool.func(reviews)

# RAG-based Q&A
rag_input = {
    "query": "What size is the bed?",
    "corpus": "This apartment features a queen-sized bed with premium linens..."
}
answer = RAGTool.func(rag_input)
```

## Configuration

Update `src/config.py` to customize:

- Data file paths
- Supported cities
- Model settings
- Temperature values

## Architecture

### Agent Flow (LCEL Chain)

1. **Extraction Step**: Parallel extraction of city and date from user query
2. **Validation Step**: Check for missing information
3. **Branching**: Either ask for clarification or search for listings
4. **Response**: Return results or clarification questions

### Tools

- **ListingSearchTool**: Search Airbnb listings by city
- **ReviewSummarizerTool**: Analyze and summarize reviews
- **RAGTool**: Answer questions using retrieval-augmented generation

### Utilities

- **NLP Utils**: Entity extraction using spaCy and LLM normalization
- **Data Loader**: Parquet file loading with error handling
- **Date Utils**: Timezone-aware date handling

## Testing

Run tests:
```bash
python -m pytest tests/
```

Or run individual test files:
```bash
python tests/test_agent.py
```

## Development

### Adding New Cities

1. Add city data files to your data directory
2. Update `SUPPORTED_CITIES` in `src/config.py`
3. Update path functions if needed

### Adding New Tools

1. Create tool function in appropriate `src/tools/` file
2. Create LangChain Tool wrapper
3. Export in `src/tools/__init__.py`
4. Integrate into agent if needed

### Code Style

Use black for formatting:
```bash
black src/ tests/ examples/
```

## Troubleshooting

### Common Issues

1. **Ollama not running**: Make sure Ollama is installed and the `llama3.2` model is available
2. **Data path errors**: Update paths in `src/config.py` to match your data location
3. **spaCy model missing**: Run `python -m spacy download en_core_web_sm`

### Error Handling

The agent includes comprehensive error handling:
- Missing data files
- LLM connection issues
- Invalid queries
- Empty results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 