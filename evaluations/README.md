# AirAgent RAG Evaluation Framework

A comprehensive evaluation system for testing RAG (Retrieval-Augmented Generation) performance using multiple LLM judges and traditional metrics.

## Overview

This framework evaluates RAG responses using both traditional metrics (word overlap, contains expected) and advanced LLM-as-judge approaches with multiple models for cross-validation.

## Framework Structure

```
evaluations/
├── datasets/rag_evaluation/     # Generated evaluation datasets
├── results/rag_evaluation/      # Evaluation results and reports  
├── utils/
│   ├── dataset_generator.py     # Create chunks and queries from listings
│   ├── llm_judge.py            # Llama 3.2 judge implementation
│   └── llm_judge_phi4.py       # Phi-4 judge implementation
├── runners/
│   └── rag_evaluator.py        # Main evaluation orchestration
└── notebooks/
    ├── dataset_creation.ipynb   # Dataset creation and exploration
    └── rag_evaluation_analysis.ipynb  # Results analysis
```

## Evaluation Methodology

### Traditional Metrics
- **Word Overlap Score**: Intersection of expected vs actual answer words
- **Contains Expected**: Boolean check if expected answer appears in response
- **Response Length**: Character count analysis
- **Error Detection**: Automatic detection of failed responses

### LLM-as-Judge Metrics
Four key dimensions scored 0.0-1.0:
- **Relevance**: How well the answer addresses the specific question
- **Completeness**: How thoroughly the answer covers user needs
- **Intent**: How well the answer fulfills underlying user intent
- **Correctness**: Factual accuracy and truthfulness

## LLM Judge Evaluation Prompts

Both our Llama 3.2 and Phi-4 judges use structured prompts to evaluate RAG responses across four dimensions. Here's what each metric measures:

### 📊 Evaluation Metrics Explained

#### **Relevance (0.0-1.0)**
*"How well the answer addresses the specific question asked"*

**What it measures**: Direct alignment between the user's question and the response content.

**Examples**:
- ✅ **High (0.9)**: User asks "How many bedrooms?" → Answer: "This property has 3 bedrooms"
- ⚠️ **Medium (0.5)**: User asks "How many bedrooms?" → Answer discusses property amenities including "spacious bedrooms"
- ❌ **Low (0.1)**: User asks "How many bedrooms?" → Answer only discusses location and pricing

#### **Completeness (0.0-1.0)**
*"How thoroughly the answer covers what the user needs to know"*

**What it measures**: Whether the response provides comprehensive information that would satisfy the user's information needs.

**Examples**:
- ✅ **High (0.9)**: User asks about amenities → Answer lists all major amenities (WiFi, kitchen, parking, pool, etc.)
- ⚠️ **Medium (0.5)**: User asks about amenities → Answer mentions some amenities but misses important ones
- ❌ **Low (0.1)**: User asks about amenities → Answer mentions only one amenity or gives vague response

#### **Intent (0.0-1.0)**
*"How well the answer fulfills the user's underlying intent and information need"*

**What it measures**: Understanding of the user's deeper purpose behind the question, beyond literal interpretation.

**Examples**:
- ✅ **High (0.9)**: User asks "Is it good for families?" → Answer discusses kid-friendly amenities, safety, nearby activities, space
- ⚠️ **Medium (0.5)**: User asks "Is it good for families?" → Answer mentions some family aspects but misses key considerations
- ❌ **Low (0.1)**: User asks "Is it good for families?" → Answer just says "yes" or discusses unrelated features

#### **Correctness (0.0-1.0)**
*"How factually accurate and truthful the answer appears to be"*

**What it measures**: Factual accuracy based on the provided context and absence of hallucinations or contradictions.

**Examples**:
- ✅ **High (0.9)**: All stated facts match the listing data, no contradictions
- ⚠️ **Medium (0.5)**: Mostly accurate but contains minor inaccuracies or unsupported claims
- ❌ **Low (0.1)**: Contains significant factual errors, contradictions, or hallucinated information

### 🎯 Judge Prompt Structure

Both judges use similar structured prompts with these key components:

1. **Context Setup**: "You are an expert evaluator with knowledge of Airbnb listings..."
2. **Input Format**: Question, Expected Answer, Actual Answer
3. **Metric Definitions**: Clear explanation of each 4 metrics
4. **Scoring Guidelines**: 0.0-1.0 scale with anchoring examples
5. **Output Format**: Strict JSON structure for consistent parsing

### 🔍 Judge Differences

**Llama 3.2 Judge Prompt**:
- Emphasizes being "wise and insightful expert"
- Focus on "everything the user would benefit from knowing"
- Instructions are more conversational

**Phi-4 Judge Prompt**:
- Emphasizes "objective evaluation" 
- Provides explicit 0.0/0.5/1.0 anchoring examples
- More structured evaluation guidelines
- Tends to be more conservative in scoring

## Multi-Judge Comparison Results

### Current RAG System: Llama 3.2 Generation

We evaluated the same RAG responses using two different LLM judges to assess inter-judge reliability and scoring differences:

#### 🧠 Llama 3.2 Judge Results
```
Success Rate: 100.0%
Relevance:    0.840
Completeness: 0.660
Intent:       0.810
Correctness:  0.900
Overall:      0.802
```

#### 🔬 Phi-4 Judge Results  
```
Success Rate: 100.0%
Relevance:    0.833
Completeness: 0.500
Intent:       0.667
Correctness:  0.900
Overall:      0.725
```

### Key Findings

**Judge Agreement:**
- **High Agreement**: Correctness scores identical (0.900)
- **Moderate Agreement**: Relevance scores very close (0.840 vs 0.833)
- **Significant Differences**: 
  - Completeness: Llama 3.2 more lenient (+0.160)
  - Intent: Llama 3.2 more generous (+0.143)
  - Overall: 0.077 point difference (Llama 3.2 higher)

**Performance Characteristics:**
- **Speed**: Llama 3.2 is ~3x faster than Phi-4 on M2 machines
- **Scoring Tendency**: Phi-4 appears more conservative in scoring
- **Consistency**: Both judges show 100% success rate in evaluation

## Usage

### Quick Start
```python
# 1. Generate evaluation dataset
from evaluations.utils.dataset_generator import DatasetGenerator
generator = DatasetGenerator()
chunks_file, queries_file = generator.create_rag_evaluation_dataset(
    city="seattle", num_listings=100
)

# 2. Run evaluation with multiple judges
from evaluations.runners.rag_evaluator import RAGEvaluator
evaluator = RAGEvaluator()

# Traditional + Llama 3.2 judge
results_llm = evaluator.run_evaluation(
    chunks_file, queries_file, use_llm_judge=True
)

# Manual Phi-4 judge comparison
from evaluations.utils.llm_judge_phi4 import LLMJudgePhi4
phi4_judge = LLMJudgePhi4()
# ... (see notebooks for detailed usage)
```

### Analysis Functions
```python
# Quick summaries
quick_judge_summary(llm_results_file)      # Llama 3.2 results
quick_phi4_summary(phi4_results_file)      # Phi-4 results
```

## Roadmap

### Planned Evaluations

1. **✅ Current**: Llama 3.2 RAG + Multi-Judge Evaluation
   - Llama 3.2 generates responses
   - Both Llama 3.2 and Phi-4 judge the same responses

2. **🔄 Next Phase**: Phi-4 RAG Generation
   - Implement Phi-4-powered RAG system
   - Compare generation quality: Llama 3.2 vs Phi-4

3. **🎯 Future**: Cross-Model Evaluation Matrix
   ```
   RAG Generator | Judge Model | Performance
   --------------|-------------|------------
   Llama 3.2     | Llama 3.2   | ✅ Baseline
   Llama 3.2     | Phi-4       | ✅ Current
   Phi-4         | Llama 3.2   | 🔄 Planned  
   Phi-4         | Phi-4       | 🔄 Planned
   ```

### Research Questions
- Does judge model bias affect evaluation when using same model for generation?
- How do cross-model evaluations compare to same-model evaluations?
- What's the optimal speed vs accuracy tradeoff for production use?

## Dependencies

- **Core**: LangChain, Ollama, pandas, numpy
- **Models**: Llama 3.2, Phi-4 (via Ollama)
- **Visualization**: matplotlib, seaborn
- **Data**: Seattle/SF Airbnb listings

## Performance Notes

**Hardware**: M2 MacBook results
- **Llama 3.2**: ~3x faster evaluation speed
- **Phi-4**: More thorough but slower evaluation
- **Memory**: Both models run efficiently on 16GB+ M2 systems

---

*This evaluation framework enables systematic comparison of RAG systems and LLM judges, providing insights into model performance, bias, and practical deployment considerations.* 